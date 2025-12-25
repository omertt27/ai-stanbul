"""
Smart Feedback Processor
========================
Intelligent processing of user feedback (👍/👎) combined with behavioral signals.

Key Principles:
1. Treat 👍/👎 as weak signals, not truth
2. Behavioral signals always dominate
3. Only trust thumbs when confirmed by behavior
4. Auto-bucket into Gold/Fail/Noise categories
5. Generate preference pairs for DPO training

Author: AI Istanbul Team
Date: December 2024
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import threading

logger = logging.getLogger(__name__)

# ============================================
# PRODUCTION-READY DATA DIRECTORY CONFIGURATION
# ============================================
# Use environment variable for production, fallback to local for development
_data_dir_path = os.environ.get('FEEDBACK_DATA_DIR', 'training_data')

# Handle both absolute and relative paths
if os.path.isabs(_data_dir_path):
    DATA_DIR = Path(_data_dir_path)
else:
    # Relative to the backend directory
    DATA_DIR = Path(__file__).parent.parent / _data_dir_path

# Create directory with proper permissions
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Feedback data directory: {DATA_DIR.absolute()}")
except Exception as e:
    logger.error(f"❌ Failed to create data directory: {e}")
    # Fallback to temp directory in production if main directory fails
    import tempfile
    DATA_DIR = Path(tempfile.gettempdir()) / "ai_istanbul_feedback"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.warning(f"⚠️ Using fallback temp directory: {DATA_DIR}")

FEEDBACK_FILE = DATA_DIR / "user_feedback.jsonl"
BEHAVIOR_FILE = DATA_DIR / "user_behavior.jsonl"
GOLD_BUCKET_FILE = DATA_DIR / "gold_examples.jsonl"
FAIL_BUCKET_FILE = DATA_DIR / "fail_examples.jsonl"
PREFERENCE_PAIRS_FILE = DATA_DIR / "preference_pairs.jsonl"
FEEDBACK_ANALYSIS_FILE = DATA_DIR / "feedback_analysis.json"


class FeedbackQuality(Enum):
    """Quality bucket for feedback"""
    GOLD = "gold"      # Behavior-positive + 👍
    FAIL = "fail"      # Behavior-negative + 👎
    NOISE = "noise"    # Everything else - not actionable


class DislikeReason(Enum):
    """One-click reasons for 👎 feedback"""
    INCORRECT = "incorrect"
    UNCLEAR = "unclear"
    TOO_GENERIC = "too_generic"
    OUTDATED = "outdated"
    OFF_TOPIC = "off_topic"
    TOO_LONG = "too_long"
    TOO_SHORT = "too_short"
    NONE = "none"


@dataclass
class BehavioralSignals:
    """Behavioral signals that indicate true quality"""
    session_continued: bool = False          # User continued the session
    rephrase_within_30s: bool = False        # User rephrased quickly (negative)
    rephrase_within_10s: bool = False        # User rephrased very quickly (strong negative)
    contradiction_detected: bool = False     # User contradicted the answer
    tool_or_link_used: bool = False          # User clicked a link or used suggested action
    time_on_response_seconds: float = 0.0    # How long user spent reading
    follow_up_question: bool = False         # User asked a related follow-up
    copied_response: bool = False            # User copied the response text
    shared_response: bool = False            # User shared the response


@dataclass
class FeedbackEntry:
    """Complete feedback entry with behavioral context"""
    interaction_id: str
    timestamp: str
    user_query: str
    llm_response: str
    
    # Explicit feedback (weak signal)
    thumbs_up: bool = False
    thumbs_down: bool = False
    dislike_reason: str = DislikeReason.NONE.value
    
    # Behavioral signals (strong signal)
    behavior: Optional[Dict[str, Any]] = None
    
    # Computed scores
    base_quality_score: float = 0.5
    adjusted_score: float = 0.5
    quality_bucket: str = FeedbackQuality.NOISE.value
    
    # Metadata
    language: str = "en"
    intent: str = "general"
    session_id: Optional[str] = None


class SmartFeedbackProcessor:
    """
    Processes user feedback intelligently following best practices:
    - Behavioral signals dominate
    - 👍/👎 only adjusts confidence
    - Auto-buckets into Gold/Fail/Noise
    - Generates preference pairs for DPO training
    """
    
    # Score adjustments
    THUMBS_UP_BONUS = 0.3
    THUMBS_DOWN_PENALTY = 0.5
    
    # Behavioral score impacts
    BEHAVIOR_WEIGHTS = {
        'session_continued': 0.2,
        'tool_or_link_used': 0.3,
        'follow_up_question': 0.15,
        'copied_response': 0.2,
        'shared_response': 0.25,
        'rephrase_within_30s': -0.3,
        'rephrase_within_10s': -0.5,
        'contradiction_detected': -0.4,
    }
    
    # Time-based scoring
    MIN_READ_TIME_SECONDS = 3.0    # Too fast = didn't read
    IDEAL_READ_TIME_SECONDS = 15.0  # Good engagement
    
    # Bucket thresholds
    GOLD_THRESHOLD = 0.7
    FAIL_THRESHOLD = 0.3
    
    def __init__(self):
        self._lock = threading.Lock()  # Thread-safe file operations
        self.analysis = self._load_analysis()
    
    def _load_analysis(self) -> Dict[str, Any]:
        """Load feedback analysis state with error handling"""
        try:
            if FEEDBACK_ANALYSIS_FILE.exists():
                with open(FEEDBACK_ANALYSIS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"⚠️ Could not load analysis file, starting fresh: {e}")
        
        return {
            'total_processed': 0,
            'gold_count': 0,
            'fail_count': 0,
            'noise_count': 0,
            'preference_pairs_generated': 0,
            'dislike_reasons': {},
            'intent_patterns': {},
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_analysis(self):
        """Save feedback analysis state with thread-safety"""
        try:
            with self._lock:
                self.analysis['last_updated'] = datetime.now().isoformat()
                # Write to temp file first, then rename (atomic operation)
                temp_file = FEEDBACK_ANALYSIS_FILE.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.analysis, f, indent=2, ensure_ascii=False)
                temp_file.replace(FEEDBACK_ANALYSIS_FILE)
        except Exception as e:
            logger.error(f"❌ Failed to save analysis: {e}")
    
    def calculate_base_quality_score(self, behavior: BehavioralSignals) -> float:
        """
        Calculate base quality score from behavioral signals only.
        This is the PRIMARY signal - thumbs just adjust it.
        """
        score = 0.5  # Start neutral
        
        # Positive behavioral signals
        if behavior.session_continued:
            score += self.BEHAVIOR_WEIGHTS['session_continued']
        if behavior.tool_or_link_used:
            score += self.BEHAVIOR_WEIGHTS['tool_or_link_used']
        if behavior.follow_up_question:
            score += self.BEHAVIOR_WEIGHTS['follow_up_question']
        if behavior.copied_response:
            score += self.BEHAVIOR_WEIGHTS['copied_response']
        if behavior.shared_response:
            score += self.BEHAVIOR_WEIGHTS['shared_response']
        
        # Negative behavioral signals
        if behavior.rephrase_within_10s:
            score += self.BEHAVIOR_WEIGHTS['rephrase_within_10s']
        elif behavior.rephrase_within_30s:
            score += self.BEHAVIOR_WEIGHTS['rephrase_within_30s']
        if behavior.contradiction_detected:
            score += self.BEHAVIOR_WEIGHTS['contradiction_detected']
        
        # Time-based adjustment
        if behavior.time_on_response_seconds > 0:
            if behavior.time_on_response_seconds < self.MIN_READ_TIME_SECONDS:
                score -= 0.1  # Too fast, probably didn't read
            elif behavior.time_on_response_seconds >= self.IDEAL_READ_TIME_SECONDS:
                score += 0.1  # Good engagement
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def adjust_score_with_thumbs(
        self, 
        base_score: float, 
        thumbs_up: bool, 
        thumbs_down: bool
    ) -> float:
        """
        Adjust score with thumbs feedback (weak signal).
        Behavioral signals always dominate.
        """
        score = base_score
        
        if thumbs_up:
            score += self.THUMBS_UP_BONUS
        if thumbs_down:
            score -= self.THUMBS_DOWN_PENALTY
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def is_feedback_trustworthy(
        self, 
        thumbs_up: bool, 
        thumbs_down: bool,
        behavior: BehavioralSignals
    ) -> Tuple[bool, str]:
        """
        Determine if thumbs feedback is trustworthy based on behavioral confirmation.
        
        Returns:
            (is_trustworthy, reason)
        """
        # Strong Positive: 👍 + No rephrase + Session continues/tool used
        if thumbs_up:
            if not behavior.rephrase_within_30s and (
                behavior.session_continued or behavior.tool_or_link_used
            ):
                return True, "thumbs_up_confirmed_by_behavior"
            return False, "thumbs_up_not_confirmed"
        
        # Strong Negative: 👎 + Rephrase within 10s OR contradiction
        if thumbs_down:
            if behavior.rephrase_within_10s or behavior.contradiction_detected:
                return True, "thumbs_down_confirmed_by_behavior"
            return False, "thumbs_down_not_confirmed"
        
        return False, "no_explicit_feedback"
    
    def determine_quality_bucket(
        self,
        adjusted_score: float,
        thumbs_up: bool,
        thumbs_down: bool,
        behavior: BehavioralSignals
    ) -> FeedbackQuality:
        """
        Determine which bucket this feedback belongs to.
        
        🟢 Gold: Behavior-positive + 👍
        🔴 Fail: Behavior-negative + 👎  
        🟡 Noise: Everything else (not actionable)
        """
        is_trustworthy, reason = self.is_feedback_trustworthy(
            thumbs_up, thumbs_down, behavior
        )
        
        # Only bucket if feedback is confirmed by behavior
        if is_trustworthy:
            if thumbs_up and adjusted_score >= self.GOLD_THRESHOLD:
                return FeedbackQuality.GOLD
            if thumbs_down and adjusted_score <= self.FAIL_THRESHOLD:
                return FeedbackQuality.FAIL
        
        # Everything else is noise
        return FeedbackQuality.NOISE
    
    def process_feedback(
        self,
        interaction_id: str,
        user_query: str,
        llm_response: str,
        thumbs_up: bool = False,
        thumbs_down: bool = False,
        dislike_reason: str = DislikeReason.NONE.value,
        behavior: Optional[BehavioralSignals] = None,
        language: str = "en",
        intent: str = "general",
        session_id: Optional[str] = None
    ) -> FeedbackEntry:
        """
        Process a feedback entry with full behavioral context.
        
        This is the main entry point for the feedback system.
        """
        behavior = behavior or BehavioralSignals()
        
        # 1. Calculate base score from behavior (primary signal)
        base_score = self.calculate_base_quality_score(behavior)
        
        # 2. Adjust with thumbs (secondary signal)
        adjusted_score = self.adjust_score_with_thumbs(
            base_score, thumbs_up, thumbs_down
        )
        
        # 3. Determine quality bucket
        quality_bucket = self.determine_quality_bucket(
            adjusted_score, thumbs_up, thumbs_down, behavior
        )
        
        # 4. Create feedback entry
        entry = FeedbackEntry(
            interaction_id=interaction_id,
            timestamp=datetime.now().isoformat(),
            user_query=user_query,
            llm_response=llm_response,
            thumbs_up=thumbs_up,
            thumbs_down=thumbs_down,
            dislike_reason=dislike_reason,
            behavior=asdict(behavior),
            base_quality_score=base_score,
            adjusted_score=adjusted_score,
            quality_bucket=quality_bucket.value,
            language=language,
            intent=intent,
            session_id=session_id
        )
        
        # 5. Save to appropriate bucket file
        self._save_to_bucket(entry, quality_bucket)
        
        # 6. Update analysis
        self._update_analysis(entry, quality_bucket, dislike_reason)
        
        logger.info(
            f"📊 Feedback processed: {interaction_id[:8]}... "
            f"Base={base_score:.2f} Adj={adjusted_score:.2f} "
            f"Bucket={quality_bucket.value}"
        )
        
        return entry
    
    def _save_to_bucket(self, entry: FeedbackEntry, bucket: FeedbackQuality):
        """Save entry to the appropriate bucket file with thread-safety and error handling"""
        entry_dict = asdict(entry)
        entry_json = json.dumps(entry_dict, ensure_ascii=False) + '\n'
        
        try:
            with self._lock:
                if bucket == FeedbackQuality.GOLD:
                    with open(GOLD_BUCKET_FILE, 'a', encoding='utf-8') as f:
                        f.write(entry_json)
                elif bucket == FeedbackQuality.FAIL:
                    with open(FAIL_BUCKET_FILE, 'a', encoding='utf-8') as f:
                        f.write(entry_json)
                
                # Always save to main feedback file
                with open(FEEDBACK_FILE, 'a', encoding='utf-8') as f:
                    f.write(entry_json)
        except IOError as e:
            logger.error(f"❌ Failed to save feedback to file: {e}")
            # Don't raise - feedback collection shouldn't break the app
    
    def _update_analysis(
        self, 
        entry: FeedbackEntry, 
        bucket: FeedbackQuality,
        dislike_reason: str
    ):
        """Update analysis statistics"""
        self.analysis['total_processed'] += 1
        
        if bucket == FeedbackQuality.GOLD:
            self.analysis['gold_count'] += 1
        elif bucket == FeedbackQuality.FAIL:
            self.analysis['fail_count'] += 1
        else:
            self.analysis['noise_count'] += 1
        
        # Track dislike reasons
        if dislike_reason and dislike_reason != DislikeReason.NONE.value:
            self.analysis['dislike_reasons'][dislike_reason] = \
                self.analysis['dislike_reasons'].get(dislike_reason, 0) + 1
        
        # Track intent patterns for failures
        if bucket == FeedbackQuality.FAIL:
            intent = entry.intent or 'unknown'
            if intent not in self.analysis['intent_patterns']:
                self.analysis['intent_patterns'][intent] = {'fail': 0, 'gold': 0}
            self.analysis['intent_patterns'][intent]['fail'] += 1
        elif bucket == FeedbackQuality.GOLD:
            intent = entry.intent or 'unknown'
            if intent not in self.analysis['intent_patterns']:
                self.analysis['intent_patterns'][intent] = {'fail': 0, 'gold': 0}
            self.analysis['intent_patterns'][intent]['gold'] += 1
        
        self._save_analysis()
    
    def generate_preference_pairs(self) -> int:
        """
        Generate preference pairs for DPO training.
        
        Creates pairs of (prompt, chosen, rejected) where:
        - chosen = 👍 high-score answer from Gold bucket
        - rejected = 👎 low-score answer from Fail bucket
        - Same prompt pattern
        
        Returns number of pairs generated.
        """
        # Load gold and fail examples
        gold_examples = []
        fail_examples = []
        
        if GOLD_BUCKET_FILE.exists():
            with open(GOLD_BUCKET_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    gold_examples.append(json.loads(line))
        
        if FAIL_BUCKET_FILE.exists():
            with open(FAIL_BUCKET_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    fail_examples.append(json.loads(line))
        
        if not gold_examples or not fail_examples:
            logger.info("Not enough examples for preference pairs")
            return 0
        
        # Group by intent and language for matching
        def get_pattern_key(example):
            return f"{example.get('intent', 'general')}_{example.get('language', 'en')}"
        
        gold_by_pattern = {}
        for ex in gold_examples:
            key = get_pattern_key(ex)
            if key not in gold_by_pattern:
                gold_by_pattern[key] = []
            gold_by_pattern[key].append(ex)
        
        fail_by_pattern = {}
        for ex in fail_examples:
            key = get_pattern_key(ex)
            if key not in fail_by_pattern:
                fail_by_pattern[key] = []
            fail_by_pattern[key].append(ex)
        
        # Generate preference pairs
        pairs_generated = 0
        
        with open(PREFERENCE_PAIRS_FILE, 'w', encoding='utf-8') as f:
            for pattern, gold_list in gold_by_pattern.items():
                if pattern not in fail_by_pattern:
                    continue
                
                fail_list = fail_by_pattern[pattern]
                
                # Create pairs (use highest scored gold, lowest scored fail)
                gold_list.sort(key=lambda x: x.get('adjusted_score', 0), reverse=True)
                fail_list.sort(key=lambda x: x.get('adjusted_score', 1))
                
                for gold_ex in gold_list[:5]:  # Top 5 gold
                    for fail_ex in fail_list[:5]:  # Worst 5 fail
                        pair = {
                            'prompt': gold_ex['user_query'],  # Use gold query as reference
                            'chosen': gold_ex['llm_response'],
                            'rejected': fail_ex['llm_response'],
                            'pattern': pattern,
                            'chosen_score': gold_ex.get('adjusted_score', 0),
                            'rejected_score': fail_ex.get('adjusted_score', 0),
                            'metadata': {
                                'gold_interaction_id': gold_ex['interaction_id'],
                                'fail_interaction_id': fail_ex['interaction_id'],
                                'gold_query': gold_ex['user_query'],
                                'fail_query': fail_ex['user_query'],
                            }
                        }
                        f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                        pairs_generated += 1
        
        self.analysis['preference_pairs_generated'] = pairs_generated
        self._save_analysis()
        
        logger.info(f"✅ Generated {pairs_generated} preference pairs for DPO training")
        return pairs_generated
    
    def get_actionable_insights(self) -> Dict[str, Any]:
        """
        Get actionable insights from feedback analysis.
        
        Identifies:
        - Intents with high failure rates (need prompt/RAG fixes)
        - Common dislike reasons (need specific improvements)
        - Patterns that need attention
        """
        insights = {
            'summary': self.analysis,
            'recommendations': [],
            'high_failure_intents': [],
            'common_issues': []
        }
        
        # Find intents with high failure rates
        for intent, counts in self.analysis.get('intent_patterns', {}).items():
            total = counts['gold'] + counts['fail']
            if total >= 5:  # Minimum samples
                fail_rate = counts['fail'] / total
                if fail_rate > 0.3:  # >30% failure
                    insights['high_failure_intents'].append({
                        'intent': intent,
                        'fail_rate': f"{fail_rate*100:.1f}%",
                        'gold': counts['gold'],
                        'fail': counts['fail'],
                        'recommendation': f"Review RAG retrieval and prompts for '{intent}' intent"
                    })
        
        # Analyze dislike reasons
        dislike_counts = self.analysis.get('dislike_reasons', {})
        total_dislikes = sum(dislike_counts.values()) or 1
        
        for reason, count in sorted(dislike_counts.items(), key=lambda x: -x[1]):
            percentage = count / total_dislikes * 100
            if percentage > 10:  # >10% of dislikes
                insights['common_issues'].append({
                    'reason': reason,
                    'count': count,
                    'percentage': f"{percentage:.1f}%"
                })
                
                # Add specific recommendations
                if reason == 'incorrect':
                    insights['recommendations'].append(
                        "High 'incorrect' feedback - Review RAG accuracy and fact-checking"
                    )
                elif reason == 'unclear':
                    insights['recommendations'].append(
                        "High 'unclear' feedback - Improve response structure and formatting"
                    )
                elif reason == 'too_generic':
                    insights['recommendations'].append(
                        "High 'too_generic' feedback - Add more specific Istanbul knowledge to prompts"
                    )
                elif reason == 'outdated':
                    insights['recommendations'].append(
                        "High 'outdated' feedback - Update RAG database with current information"
                    )
        
        return insights
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current feedback statistics"""
        return {
            'total_processed': self.analysis['total_processed'],
            'gold_count': self.analysis['gold_count'],
            'fail_count': self.analysis['fail_count'],
            'noise_count': self.analysis['noise_count'],
            'gold_rate': (
                self.analysis['gold_count'] / max(self.analysis['total_processed'], 1) * 100
            ),
            'fail_rate': (
                self.analysis['fail_count'] / max(self.analysis['total_processed'], 1) * 100
            ),
            'preference_pairs_generated': self.analysis['preference_pairs_generated'],
            'dislike_reasons': self.analysis.get('dislike_reasons', {}),
            'last_updated': self.analysis.get('last_updated')
        }


# Global instance
smart_feedback_processor = SmartFeedbackProcessor()


def process_smart_feedback(
    interaction_id: str,
    user_query: str,
    llm_response: str,
    thumbs_up: bool = False,
    thumbs_down: bool = False,
    dislike_reason: str = "none",
    **kwargs
) -> FeedbackEntry:
    """Convenience function to process feedback"""
    return smart_feedback_processor.process_feedback(
        interaction_id=interaction_id,
        user_query=user_query,
        llm_response=llm_response,
        thumbs_up=thumbs_up,
        thumbs_down=thumbs_down,
        dislike_reason=dislike_reason,
        **kwargs
    )


def get_feedback_insights() -> Dict[str, Any]:
    """Get actionable insights from feedback"""
    return smart_feedback_processor.get_actionable_insights()


def generate_dpo_pairs() -> int:
    """Generate preference pairs for DPO training"""
    return smart_feedback_processor.generate_preference_pairs()


def get_smart_feedback_stats() -> Dict[str, Any]:
    """Get feedback statistics"""
    return smart_feedback_processor.get_stats()
