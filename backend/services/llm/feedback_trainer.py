"""
feedback_trainer.py - Feedback Loop Training System

Analyzes intent discrepancies between regex and LLM detection,
generates new patterns, and manages the retraining pipeline.

Author: AI Istanbul Team
Date: December 2025
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path

logger = logging.getLogger(__name__)


class FeedbackTrainer:
    """
    Manages the feedback loop for signal detection improvement.
    
    Features:
    - Logs intent discrepancies
    - Analyzes patterns in missed detections
    - Generates new regex patterns
    - Manages retraining pipeline
    - Validates pattern quality
    """
    
    def __init__(
        self,
        training_data_path: str = "data/training_samples.jsonl",
        patterns_output_path: str = "backend/services/llm/learned_patterns.json",
        min_samples_for_retraining: int = 100,
        min_pattern_confidence: float = 0.7
    ):
        """
        Initialize feedback trainer.
        
        Args:
            training_data_path: Path to store training samples
            patterns_output_path: Path to save learned patterns
            min_samples_for_retraining: Minimum samples before suggesting retraining
            min_pattern_confidence: Minimum confidence for pattern suggestions
        """
        self.training_data_path = Path(training_data_path)
        self.patterns_output_path = Path(patterns_output_path)
        self.min_samples_for_retraining = min_samples_for_retraining
        self.min_pattern_confidence = min_pattern_confidence
        
        # Ensure directories exist
        self.training_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.patterns_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for fast access
        self.training_samples: List[Dict[str, Any]] = []
        self.intent_discrepancies: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'discrepancies_by_intent': defaultdict(int),
            'queries_by_language': defaultdict(int),
            'last_retrain': None,
            'patterns_suggested': 0
        }
        
        # Load existing training data
        self._load_training_data()
        
        logger.info(f"✅ FeedbackTrainer initialized with {len(self.training_samples)} existing samples")
    
    def _load_training_data(self):
        """Load existing training samples from disk."""
        if not self.training_data_path.exists():
            logger.info("No existing training data found")
            return
        
        try:
            with open(self.training_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        self.training_samples.append(sample)
                        
                        # Update statistics
                        self.stats['total_samples'] += 1
                        if sample.get('has_discrepancy'):
                            for intent in sample.get('missed_intents', []):
                                self.stats['discrepancies_by_intent'][intent] += 1
                        self.stats['queries_by_language'][sample.get('language', 'unknown')] += 1
            
            logger.info(f"Loaded {len(self.training_samples)} training samples")
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
    
    def log_training_sample(
        self,
        query: str,
        regex_intents: Dict[str, bool],
        llm_intents: Dict[str, bool],
        language: str = 'en',
        user_location: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a training sample for analysis.
        
        Args:
            query: The user query
            regex_intents: Intents detected by regex
            llm_intents: Intents detected by LLM
            language: Query language
            user_location: User's GPS location (if available)
            metadata: Additional metadata (confidence scores, etc.)
        """
        # Calculate discrepancies
        missed_intents = [
            intent for intent, llm_value in llm_intents.items()
            if llm_value and not regex_intents.get(intent, False)
        ]
        
        false_positives = [
            intent for intent, regex_value in regex_intents.items()
            if regex_value and not llm_intents.get(intent, False)
        ]
        
        has_discrepancy = len(missed_intents) > 0 or len(false_positives) > 0
        
        # Create training sample
        sample = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'query_lower': query.lower(),
            'language': language,
            'regex_intents': regex_intents,
            'llm_intents': llm_intents,
            'missed_intents': missed_intents,
            'false_positives': false_positives,
            'has_discrepancy': has_discrepancy,
            'user_location': user_location,
            'metadata': metadata or {}
        }
        
        # Add to in-memory cache
        self.training_samples.append(sample)
        
        # Update discrepancy tracking
        if has_discrepancy:
            for intent in missed_intents:
                self.intent_discrepancies[intent].append(sample)
                self.stats['discrepancies_by_intent'][intent] += 1
        
        # Update statistics
        self.stats['total_samples'] += 1
        self.stats['queries_by_language'][language] += 1
        
        # Persist to disk (append mode)
        self._save_sample(sample)
        
        if has_discrepancy:
            logger.info(
                f"📝 Training sample logged: query='{query[:50]}...', "
                f"missed={missed_intents}, false_positives={false_positives}"
            )
    
    def _save_sample(self, sample: Dict[str, Any]):
        """Append a single sample to the training data file."""
        try:
            with open(self.training_data_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Error saving training sample: {e}")
    
    def get_retraining_readiness(self) -> Dict[str, Any]:
        """
        Check if enough samples have been collected for retraining.
        
        Returns:
            Dict with readiness status and recommendations
        """
        total_discrepancies = sum(self.stats['discrepancies_by_intent'].values())
        
        is_ready = total_discrepancies >= self.min_samples_for_retraining
        
        return {
            'is_ready': is_ready,
            'total_samples': self.stats['total_samples'],
            'total_discrepancies': total_discrepancies,
            'min_required': self.min_samples_for_retraining,
            'discrepancies_by_intent': dict(self.stats['discrepancies_by_intent']),
            'recommendation': (
                "Ready for retraining" if is_ready
                else f"Need {self.min_samples_for_retraining - total_discrepancies} more discrepancy samples"
            )
        }
    
    def analyze_missed_patterns(
        self,
        intent: str,
        language: str = 'en',
        min_occurrences: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze queries where an intent was missed by regex but caught by LLM.
        
        Args:
            intent: The intent to analyze (e.g., 'needs_restaurant')
            language: Filter by language
            min_occurrences: Minimum times a pattern must appear
        
        Returns:
            Dict with analysis results and suggested patterns
        """
        # Get all samples where this intent was missed
        missed_samples = [
            s for s in self.intent_discrepancies[intent]
            if s.get('language') == language
        ]
        
        if len(missed_samples) < min_occurrences:
            return {
                'intent': intent,
                'language': language,
                'sample_count': len(missed_samples),
                'status': 'insufficient_data',
                'suggested_patterns': []
            }
        
        # Extract common words and phrases
        keyword_candidates = self._extract_keyword_candidates(
            missed_samples, min_occurrences
        )
        
        # Generate regex pattern suggestions
        suggested_patterns = self._generate_pattern_suggestions(
            keyword_candidates, intent, language
        )
        
        # Calculate confidence scores
        patterns_with_confidence = self._calculate_pattern_confidence(
            suggested_patterns, missed_samples
        )
        
        # Filter by minimum confidence
        high_confidence_patterns = [
            p for p in patterns_with_confidence
            if p['confidence'] >= self.min_pattern_confidence
        ]
        
        return {
            'intent': intent,
            'language': language,
            'sample_count': len(missed_samples),
            'status': 'analyzed',
            'keyword_candidates': keyword_candidates,
            'suggested_patterns': high_confidence_patterns,
            'example_queries': [s['query'] for s in missed_samples[:5]]
        }
    
    def _extract_keyword_candidates(
        self,
        samples: List[Dict[str, Any]],
        min_occurrences: int
    ) -> List[Dict[str, Any]]:
        """Extract frequently occurring words/phrases from missed queries."""
        # Common words to ignore (stopwords)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
            'can', 'could', 'will', 'would', 'should', 'may', 'might', 'must'
        }
        
        # Extract unigrams (single words)
        unigram_counter = Counter()
        for sample in samples:
            words = re.findall(r'\b\w+\b', sample['query_lower'])
            for word in words:
                if word not in stopwords and len(word) > 2:
                    unigram_counter[word] += 1
        
        # Extract bigrams (two-word phrases)
        bigram_counter = Counter()
        for sample in samples:
            words = re.findall(r'\b\w+\b', sample['query_lower'])
            for i in range(len(words) - 1):
                if words[i] not in stopwords or words[i+1] not in stopwords:
                    bigram = f"{words[i]} {words[i+1]}"
                    bigram_counter[bigram] += 1
        
        # Filter by minimum occurrences
        candidates = []
        
        for word, count in unigram_counter.items():
            if count >= min_occurrences:
                candidates.append({
                    'text': word,
                    'type': 'unigram',
                    'occurrences': count,
                    'frequency': count / len(samples)
                })
        
        for phrase, count in bigram_counter.items():
            if count >= min_occurrences:
                candidates.append({
                    'text': phrase,
                    'type': 'bigram',
                    'occurrences': count,
                    'frequency': count / len(samples)
                })
        
        # Sort by occurrences
        candidates.sort(key=lambda x: x['occurrences'], reverse=True)
        
        return candidates[:20]  # Top 20 candidates
    
    def _generate_pattern_suggestions(
        self,
        candidates: List[Dict[str, Any]],
        intent: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """Generate regex pattern suggestions from keyword candidates."""
        patterns = []
        
        for candidate in candidates:
            text = candidate['text']
            pattern_type = candidate['type']
            
            if pattern_type == 'unigram':
                # Simple word boundary pattern
                regex = rf'\b{re.escape(text)}\b'
                patterns.append({
                    'pattern': regex,
                    'description': f"Matches word '{text}'",
                    'example_match': text,
                    'candidate': candidate
                })
                
                # Add plural/variant patterns if applicable
                if not text.endswith('s'):
                    plural_regex = rf'\b{re.escape(text)}s?\b'
                    patterns.append({
                        'pattern': plural_regex,
                        'description': f"Matches '{text}' or '{text}s'",
                        'example_match': f"{text}/{text}s",
                        'candidate': candidate
                    })
            
            elif pattern_type == 'bigram':
                # Phrase pattern with optional words in between
                words = text.split()
                regex = r'\b' + re.escape(words[0]) + r'\s+' + re.escape(words[1]) + r'\b'
                patterns.append({
                    'pattern': regex,
                    'description': f"Matches phrase '{text}'",
                    'example_match': text,
                    'candidate': candidate
                })
        
        return patterns
    
    def _calculate_pattern_confidence(
        self,
        patterns: List[Dict[str, Any]],
        samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calculate confidence score for each pattern based on how well it matches samples."""
        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                
                # Count matches in missed samples (true positives)
                matches = sum(
                    1 for sample in samples
                    if compiled.search(sample['query'])
                )
                
                # Confidence = match rate in discrepancy samples
                confidence = matches / len(samples) if len(samples) > 0 else 0.0
                
                pattern_info['confidence'] = confidence
                pattern_info['matches'] = matches
                pattern_info['total_samples'] = len(samples)
                
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                pattern_info['confidence'] = 0.0
                pattern_info['matches'] = 0
                pattern_info['total_samples'] = len(samples)
        
        # Sort by confidence
        patterns.sort(key=lambda x: x['confidence'], reverse=True)
        
        return patterns
    
    def generate_retraining_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report for retraining.
        
        Returns:
            Dict with analysis for all intents and suggested patterns
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'statistics': dict(self.stats),
            'readiness': self.get_retraining_readiness(),
            'intent_analyses': {}
        }
        
        # Analyze each intent with discrepancies
        for intent in self.intent_discrepancies.keys():
            for language in ['en', 'tr', 'ar']:  # Primary languages
                analysis = self.analyze_missed_patterns(intent, language)
                
                if analysis['status'] == 'analyzed' and analysis['suggested_patterns']:
                    key = f"{intent}_{language}"
                    report['intent_analyses'][key] = analysis
        
        # Save report to file
        report_path = self.patterns_output_path.parent / 'retraining_report.json'
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"📊 Retraining report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
        
        return report
    
    def export_learned_patterns(self) -> bool:
        """
        Export learned patterns in format compatible with signals.py.
        
        Returns:
            True if export successful
        """
        report = self.generate_retraining_report()
        
        if not report['readiness']['is_ready']:
            logger.warning("Not enough samples for pattern export")
            return False
        
        # Format patterns for signals.py
        learned_patterns = {
            'version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'source': 'feedback_loop',
            'patterns': {}
        }
        
        for intent_lang, analysis in report['intent_analyses'].items():
            intent = analysis['intent']
            language = analysis['language']
            
            if intent not in learned_patterns['patterns']:
                learned_patterns['patterns'][intent] = {}
            
            if language not in learned_patterns['patterns'][intent]:
                learned_patterns['patterns'][intent][language] = []
            
            # Add high-confidence patterns
            for pattern_info in analysis['suggested_patterns'][:5]:  # Top 5
                learned_patterns['patterns'][intent][language].append({
                    'pattern': pattern_info['pattern'],
                    'confidence': pattern_info['confidence'],
                    'description': pattern_info['description'],
                    'source': 'feedback_loop'
                })
        
        # Save to file
        try:
            with open(self.patterns_output_path, 'w', encoding='utf-8') as f:
                json.dump(learned_patterns, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Learned patterns exported to {self.patterns_output_path}")
            self.stats['patterns_suggested'] = sum(
                len(patterns) 
                for intent_patterns in learned_patterns['patterns'].values()
                for patterns in intent_patterns.values()
            )
            self.stats['last_retrain'] = datetime.now().isoformat()
            
            return True
        except Exception as e:
            logger.error(f"Error exporting patterns: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        readiness = self.get_retraining_readiness()
        
        return {
            **self.stats,
            'readiness': readiness,
            'recent_samples': len([
                s for s in self.training_samples
                if datetime.fromisoformat(s['timestamp']) > datetime.now() - timedelta(days=7)
            ])
        }
