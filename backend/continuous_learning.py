#!/usr/bin/env python3
"""
Continuous Learning & Feedback System
====================================

Learns from user interactions and feedback to improve query understanding
without GPT/LLM dependencies. Uses pattern recognition and statistical learning.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import os

@dataclass
class QueryFeedback:
    """User feedback on query understanding"""
    query: str
    predicted_intent: str
    predicted_entities: Dict[str, Any]
    actual_intent: Optional[str]
    actual_entities: Optional[Dict[str, Any]]
    user_satisfaction: int  # 1-5 scale
    timestamp: str
    correction_type: str  # 'intent', 'entities', 'both', 'none'

@dataclass
class LearningPattern:
    """Learned pattern from user feedback"""
    pattern_text: str
    intent: str
    entities: Dict[str, Any]
    confidence: float
    usage_count: int
    success_rate: float
    last_used: str

class FeedbackCollector:
    """Collects and analyzes user feedback"""
    
    def __init__(self, feedback_file: str = "query_feedback_log.json"):
        self.feedback_file = feedback_file
        self.feedback_log = []
        self.load_feedback()
        
        # Performance tracking
        self.intent_accuracy = defaultdict(list)
        self.entity_accuracy = defaultdict(list)
        self.overall_satisfaction = []
    
    def add_feedback(self, query: str, predicted_intent: str, 
                    predicted_entities: Dict[str, Any], user_feedback: Dict[str, Any]):
        """Add user feedback"""
        
        # Determine correction type
        correction_type = 'none'
        actual_intent = user_feedback.get('correct_intent')
        actual_entities = user_feedback.get('correct_entities')
        
        if actual_intent and actual_intent != predicted_intent:
            correction_type = 'intent'
        if actual_entities and actual_entities != predicted_entities:
            if correction_type == 'intent':
                correction_type = 'both'
            else:
                correction_type = 'entities'
        
        feedback = QueryFeedback(
            query=query,
            predicted_intent=predicted_intent,
            predicted_entities=predicted_entities,
            actual_intent=actual_intent,
            actual_entities=actual_entities,
            user_satisfaction=user_feedback.get('satisfaction', 3),
            timestamp=datetime.now().isoformat(),
            correction_type=correction_type
        )
        
        self.feedback_log.append(feedback)
        self.save_feedback()
        
        # Update performance tracking
        self._update_performance_metrics(feedback)
    
    def add_implicit_feedback(self, query: str, predicted_intent: str,
                            predicted_entities: Dict[str, Any], user_action: str):
        """Add implicit feedback from user actions"""
        
        # Convert user actions to satisfaction scores
        satisfaction_mapping = {
            'clicked_result': 5,  # User found what they wanted
            'refined_query': 3,   # Partially helpful
            'changed_query': 2,   # Not helpful
            'abandoned': 1        # Completely unhelpful
        }
        
        satisfaction = satisfaction_mapping.get(user_action, 3)
        
        self.add_feedback(query, predicted_intent, predicted_entities, {
            'satisfaction': satisfaction,
            'feedback_type': 'implicit',
            'user_action': user_action
        })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance analytics report"""
        if not self.feedback_log:
            return {'message': 'No feedback data available'}
        
        recent_feedback = [f for f in self.feedback_log 
                          if datetime.fromisoformat(f.timestamp) > datetime.now() - timedelta(days=7)]
        
        # Intent accuracy
        intent_correct = sum(1 for f in recent_feedback 
                           if f.correction_type in ['none', 'entities'])
        intent_accuracy = intent_correct / len(recent_feedback) if recent_feedback else 0
        
        # Entity accuracy
        entity_correct = sum(1 for f in recent_feedback 
                           if f.correction_type in ['none', 'intent'])
        entity_accuracy = entity_correct / len(recent_feedback) if recent_feedback else 0
        
        # Satisfaction
        avg_satisfaction = sum(f.user_satisfaction for f in recent_feedback) / len(recent_feedback) if recent_feedback else 0
        
        # Common issues
        issue_patterns = self._identify_common_issues(recent_feedback)
        
        return {
            'total_feedback': len(self.feedback_log),
            'recent_feedback': len(recent_feedback),
            'intent_accuracy': intent_accuracy,
            'entity_accuracy': entity_accuracy,
            'average_satisfaction': avg_satisfaction,
            'common_issues': issue_patterns,
            'improvement_suggestions': self._generate_improvement_suggestions(recent_feedback)
        }
    
    def _identify_common_issues(self, feedback_list: List[QueryFeedback]) -> List[Dict[str, Any]]:
        """Identify common patterns in misunderstood queries"""
        issues = []
        
        # Intent misclassification patterns
        intent_errors = defaultdict(list)
        for f in feedback_list:
            if f.correction_type in ['intent', 'both'] and f.actual_intent:
                intent_errors[(f.predicted_intent, f.actual_intent)].append(f.query)
        
        for (predicted, actual), queries in intent_errors.items():
            if len(queries) >= 2:  # Pattern if appears 2+ times
                issues.append({
                    'type': 'intent_misclassification',
                    'predicted': predicted,
                    'actual': actual,
                    'frequency': len(queries),
                    'example_queries': queries[:3]
                })
        
        # Entity extraction issues
        entity_issues = defaultdict(int)
        for f in feedback_list:
            if f.correction_type in ['entities', 'both']:
                # Identify which entities were missed or wrong
                predicted_districts = set(f.predicted_entities.get('districts', []))
                actual_districts = set(f.actual_entities.get('districts', []) if f.actual_entities else [])
                
                if predicted_districts != actual_districts:
                    entity_issues['district_extraction'] += 1
        
        for issue_type, count in entity_issues.items():
            if count >= 2:
                issues.append({
                    'type': 'entity_extraction',
                    'issue': issue_type,
                    'frequency': count
                })
        
        return issues
    
    def _generate_improvement_suggestions(self, feedback_list: List[QueryFeedback]) -> List[str]:
        """Generate suggestions for system improvement"""
        suggestions = []
        
        # Low satisfaction patterns
        low_satisfaction = [f for f in feedback_list if f.user_satisfaction <= 2]
        if len(low_satisfaction) > len(feedback_list) * 0.2:  # More than 20% low satisfaction
            suggestions.append("High number of low satisfaction scores - review intent classification patterns")
        
        # Intent accuracy issues
        intent_errors = sum(1 for f in feedback_list if f.correction_type in ['intent', 'both'])
        if intent_errors > len(feedback_list) * 0.3:  # More than 30% intent errors
            suggestions.append("Intent classification needs improvement - consider adding more patterns")
        
        # Entity extraction issues
        entity_errors = sum(1 for f in feedback_list if f.correction_type in ['entities', 'both'])
        if entity_errors > len(feedback_list) * 0.3:  # More than 30% entity errors
            suggestions.append("Entity extraction needs improvement - expand location/category dictionaries")
        
        return suggestions
    
    def _update_performance_metrics(self, feedback: QueryFeedback):
        """Update performance tracking metrics"""
        # Intent accuracy
        intent_correct = feedback.correction_type not in ['intent', 'both']
        self.intent_accuracy[feedback.predicted_intent].append(intent_correct)
        
        # Entity accuracy
        entity_correct = feedback.correction_type not in ['entities', 'both']
        self.entity_accuracy['overall'].append(entity_correct)
        
        # Overall satisfaction
        self.overall_satisfaction.append(feedback.user_satisfaction)
    
    def save_feedback(self):
        """Save feedback to file"""
        try:
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(fb) for fb in self.feedback_log], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save feedback: {e}")
    
    def load_feedback(self):
        """Load feedback from file"""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.feedback_log = [QueryFeedback(**item) for item in data]
        except Exception as e:
            print(f"Warning: Could not load feedback: {e}")
            self.feedback_log = []

class PatternLearner:
    """Learns new patterns from user feedback"""
    
    def __init__(self, patterns_file: str = "learned_patterns.json"):
        self.patterns_file = patterns_file
        self.learned_patterns = []
        self.load_patterns()
        
        # Pattern matching thresholds
        self.min_pattern_confidence = 0.7
        self.min_usage_count = 2
    
    def learn_from_feedback(self, feedback_list: List[QueryFeedback]):
        """Learn new patterns from feedback"""
        
        # Group corrections by pattern
        correction_patterns = defaultdict(list)
        
        for feedback in feedback_list:
            if feedback.correction_type != 'none' and feedback.actual_intent:
                # Extract pattern from query
                pattern = self._extract_pattern(feedback.query)
                correction_patterns[pattern].append({
                    'intent': feedback.actual_intent,
                    'entities': feedback.actual_entities or {},
                    'satisfaction': feedback.user_satisfaction
                })
        
        # Create learned patterns
        for pattern_text, corrections in correction_patterns.items():
            if len(corrections) >= self.min_usage_count:
                # Calculate pattern confidence
                intent_counts = Counter(c['intent'] for c in corrections)
                most_common_intent = intent_counts.most_common(1)[0]
                confidence = most_common_intent[1] / len(corrections)
                
                if confidence >= self.min_pattern_confidence:
                    # Create learned pattern
                    avg_satisfaction = sum(c['satisfaction'] for c in corrections) / len(corrections)
                    
                    learned_pattern = LearningPattern(
                        pattern_text=pattern_text,
                        intent=most_common_intent[0],
                        entities=self._merge_entities([c['entities'] for c in corrections]),
                        confidence=confidence,
                        usage_count=len(corrections),
                        success_rate=avg_satisfaction / 5.0,
                        last_used=datetime.now().isoformat()
                    )
                    
                    # Add or update pattern
                    self._add_or_update_pattern(learned_pattern)
        
        self.save_patterns()
    
    def apply_learned_patterns(self, query: str) -> Optional[Tuple[str, Dict[str, Any], float]]:
        """Apply learned patterns to improve query understanding"""
        query_lower = query.lower()
        
        # Find matching patterns
        matches = []
        for pattern in self.learned_patterns:
            if self._pattern_matches(query_lower, pattern.pattern_text):
                score = pattern.confidence * pattern.success_rate * min(pattern.usage_count / 10, 1.0)
                matches.append((pattern, score))
        
        if matches:
            # Get best matching pattern
            best_pattern, score = max(matches, key=lambda x: x[1])
            
            # Update usage
            best_pattern.usage_count += 1
            best_pattern.last_used = datetime.now().isoformat()
            
            return best_pattern.intent, best_pattern.entities, score
        
        return None
    
    def _extract_pattern(self, query: str) -> str:
        """Extract a pattern from query for learning"""
        # Normalize query
        normalized = query.lower()
        
        # Replace specific entities with placeholders
        districts = ['sultanahmet', 'beyoğlu', 'kadıköy', 'beşiktaş', 'üsküdar']
        for district in districts:
            if district in normalized:
                normalized = normalized.replace(district, '[DISTRICT]')
        
        cuisines = ['turkish', 'italian', 'japanese', 'seafood']
        for cuisine in cuisines:
            if cuisine in normalized:
                normalized = normalized.replace(cuisine, '[CUISINE]')
        
        return normalized
    
    def _pattern_matches(self, query: str, pattern: str) -> bool:
        """Check if query matches learned pattern"""
        # Simple pattern matching - can be enhanced
        pattern_words = set(pattern.replace('[DISTRICT]', '').replace('[CUISINE]', '').split())
        query_words = set(query.split())
        
        # Calculate word overlap
        overlap = len(pattern_words.intersection(query_words))
        min_words = min(len(pattern_words), len(query_words))
        
        return overlap / max(min_words, 1) > 0.6  # 60% word overlap threshold
    
    def _merge_entities(self, entity_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge entities from multiple examples"""
        merged = defaultdict(list)
        
        for entities in entity_list:
            for key, value in entities.items():
                if isinstance(value, list):
                    merged[key].extend(value)
                elif value:  # Not None or empty
                    merged[key].append(value)
        
        # Remove duplicates and return
        result = {}
        for key, values in merged.items():
            if isinstance(values[0], str):
                result[key] = list(set(values))
            else:
                result[key] = values[0] if values else None
        
        return result
    
    def _add_or_update_pattern(self, new_pattern: LearningPattern):
        """Add new pattern or update existing one"""
        for i, existing in enumerate(self.learned_patterns):
            if existing.pattern_text == new_pattern.pattern_text:
                # Update existing pattern
                existing.usage_count += new_pattern.usage_count
                existing.confidence = max(existing.confidence, new_pattern.confidence)
                existing.success_rate = (existing.success_rate + new_pattern.success_rate) / 2
                existing.last_used = new_pattern.last_used
                return
        
        # Add new pattern
        self.learned_patterns.append(new_pattern)
    
    def save_patterns(self):
        """Save learned patterns to file"""
        try:
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(p) for p in self.learned_patterns], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save patterns: {e}")
    
    def load_patterns(self):
        """Load learned patterns from file"""
        try:
            if os.path.exists(self.patterns_file):
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.learned_patterns = [LearningPattern(**item) for item in data]
        except Exception as e:
            print(f"Warning: Could not load patterns: {e}")
            self.learned_patterns = []

class ContinuousLearningSystem:
    """Main system that coordinates feedback collection and pattern learning"""
    
    def __init__(self):
        self.feedback_collector = FeedbackCollector()
        self.pattern_learner = PatternLearner()
        
        # Learning schedule
        self.last_learning_update = None
        self.learning_interval = timedelta(hours=1)  # Learn every hour
    
    def add_user_feedback(self, query: str, predicted_intent: str,
                         predicted_entities: Dict[str, Any], user_feedback: Dict[str, Any]):
        """Add user feedback and trigger learning if needed"""
        
        self.feedback_collector.add_feedback(query, predicted_intent, predicted_entities, user_feedback)
        
        # Check if it's time to learn new patterns
        if self._should_update_patterns():
            self.update_learned_patterns()
    
    def add_implicit_feedback(self, query: str, predicted_intent: str,
                            predicted_entities: Dict[str, Any], user_action: str):
        """Add implicit feedback from user behavior"""
        
        self.feedback_collector.add_implicit_feedback(query, predicted_intent, predicted_entities, user_action)
    
    def improve_query_understanding(self, query: str, session_id: str) -> Dict[str, Any]:
        """Apply learned improvements to query understanding"""
        from enhanced_query_understanding import process_enhanced_query
        
        # Get baseline understanding
        result = process_enhanced_query(query, session_id)
        
        # Apply learned patterns
        learned_result = self.pattern_learner.apply_learned_patterns(query)
        
        if learned_result:
            learned_intent, learned_entities, learned_confidence = learned_result
            
            # If learned pattern has higher confidence, use it
            if learned_confidence > result['confidence']:
                result['intent'] = learned_intent
                result['entities'] = learned_entities
                result['confidence'] = learned_confidence
                result['learned_pattern_applied'] = True
                result['learning_confidence'] = learned_confidence
            else:
                result['learned_pattern_applied'] = False
        else:
            result['learned_pattern_applied'] = False
        
        return result
    
    def update_learned_patterns(self):
        """Update learned patterns from recent feedback"""
        print("🎓 Updating learned patterns from user feedback...")
        
        # Get recent feedback
        recent_feedback = [f for f in self.feedback_collector.feedback_log
                          if datetime.fromisoformat(f.timestamp) > datetime.now() - timedelta(days=7)]
        
        # Learn new patterns
        self.pattern_learner.learn_from_feedback(recent_feedback)
        
        self.last_learning_update = datetime.now()
        print(f"✅ Pattern learning update completed. {len(self.pattern_learner.learned_patterns)} patterns learned.")
    
    def _should_update_patterns(self) -> bool:
        """Check if it's time to update learned patterns"""
        if self.last_learning_update is None:
            return True
        
        return datetime.now() - self.last_learning_update > self.learning_interval
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Get comprehensive learning and performance report"""
        feedback_report = self.feedback_collector.get_performance_report()
        
        return {
            'feedback_analytics': feedback_report,
            'learned_patterns_count': len(self.pattern_learner.learned_patterns),
            'active_patterns': [p for p in self.pattern_learner.learned_patterns 
                              if datetime.fromisoformat(p.last_used) > datetime.now() - timedelta(days=30)],
            'learning_status': {
                'last_update': self.last_learning_update.isoformat() if self.last_learning_update else None,
                'next_update': (self.last_learning_update + self.learning_interval).isoformat() 
                              if self.last_learning_update else 'On next feedback'
            }
        }

# Global learning system
continuous_learning = ContinuousLearningSystem()

if __name__ == "__main__":
    # Test continuous learning system
    print("🎓 Continuous Learning System Test")
    print("=" * 40)
    
    # Simulate user feedback
    test_feedback = [
        {
            'query': 'show me good eats in sultanahmet',
            'predicted_intent': 'general_info',
            'predicted_entities': {'districts': ['sultanahmet']},
            'user_feedback': {
                'correct_intent': 'find_restaurant',
                'correct_entities': {'districts': ['sultanahmet'], 'categories': ['restaurant']},
                'satisfaction': 4
            }
        },
        {
            'query': 'where can I grab food near beyoglu',
            'predicted_intent': 'find_attraction',
            'predicted_entities': {'districts': ['beyoglu']},
            'user_feedback': {
                'correct_intent': 'find_restaurant',
                'correct_entities': {'districts': ['beyoglu'], 'categories': ['restaurant']},
                'satisfaction': 5
            }
        }
    ]
    
    # Add feedback
    for feedback in test_feedback:
        continuous_learning.add_user_feedback(
            feedback['query'],
            feedback['predicted_intent'],
            feedback['predicted_entities'],
            feedback['user_feedback']
        )
    
    # Test improved understanding
    test_query = "show me good eats in kadikoy"
    print(f"\n🔍 Testing Query: '{test_query}'")
    print("-" * 30)
    
    result = continuous_learning.improve_query_understanding(test_query, "test_session")
    
    print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
    print(f"Learned Pattern Applied: {result.get('learned_pattern_applied', False)}")
    
    if result['entities']['districts']:
        print(f"Districts: {', '.join(result['entities']['districts'])}")
    if result['entities']['categories']:
        print(f"Categories: {', '.join(result['entities']['categories'])}")
    
    # Show learning report
    print(f"\n📊 Learning Report")
    print("-" * 20)
    report = continuous_learning.get_learning_report()
    
    feedback_analytics = report['feedback_analytics']
    print(f"Total Feedback: {feedback_analytics.get('total_feedback', 0)}")
    print(f"Intent Accuracy: {feedback_analytics.get('intent_accuracy', 0):.1%}")
    print(f"Average Satisfaction: {feedback_analytics.get('average_satisfaction', 0):.1f}/5")
    print(f"Learned Patterns: {report['learned_patterns_count']}")
    
    if report['feedback_analytics'].get('improvement_suggestions'):
        print("Improvement Suggestions:")
        for suggestion in report['feedback_analytics']['improvement_suggestions']:
            print(f"  • {suggestion}")
