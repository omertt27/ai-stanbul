#!/usr/bin/env python3
"""
Enhanced Quality Scoring System for AI Istanbul
==============================================

Integrates with content accuracy system to provide comprehensive quality assessment
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class QualityDimension(Enum):
    """Quality assessment dimensions"""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    CULTURAL_AWARENESS = "cultural_awareness"
    PRACTICAL_UTILITY = "practical_utility"
    LANGUAGE_QUALITY = "language_quality"
    SOURCE_RELIABILITY = "source_reliability"

@dataclass
class QualityScore:
    """Individual quality dimension score"""
    dimension: QualityDimension
    score: float  # 0.0 to 5.0
    explanation: str
    contributing_factors: List[str]
    improvement_suggestions: List[str]

@dataclass
class ComprehensiveQualityAssessment:
    """Complete quality assessment result"""
    overall_score: float  # 0.0 to 5.0
    letter_grade: str     # A+ to F
    dimension_scores: Dict[QualityDimension, QualityScore]
    strengths: List[str]
    weaknesses: List[str]
    improvement_priority: List[str]
    confidence_level: float
    assessment_timestamp: str

class EnhancedQualityScorer:
    """Advanced quality scoring system with multiple dimensions"""
    
    def __init__(self):
        self.scoring_weights = self._initialize_scoring_weights()
        self.quality_benchmarks = self._initialize_quality_benchmarks()
        self.cultural_sensitivity_markers = self._initialize_cultural_markers()
        
        logger.info("✅ Enhanced Quality Scorer initialized")
    
    def assess_response_quality(self, response: str, query: str, category: str, 
                               accuracy_result: Optional[Any] = None) -> ComprehensiveQualityAssessment:
        """
        Comprehensive quality assessment of AI response
        
        Args:
            response: AI response to assess
            query: Original user query
            category: Query category
            accuracy_result: Optional ContentAccuracyResult from accuracy system
            
        Returns:
            ComprehensiveQualityAssessment with detailed scoring
        """
        
        try:
            # Score each quality dimension
            dimension_scores = {}
            
            # 1. Accuracy Assessment
            dimension_scores[QualityDimension.ACCURACY] = self._assess_accuracy(
                response, query, category, accuracy_result
            )
            
            # 2. Relevance Assessment
            dimension_scores[QualityDimension.RELEVANCE] = self._assess_relevance(
                response, query, category
            )
            
            # 3. Completeness Assessment
            dimension_scores[QualityDimension.COMPLETENESS] = self._assess_completeness(
                response, query, category
            )
            
            # 4. Cultural Awareness Assessment
            dimension_scores[QualityDimension.CULTURAL_AWARENESS] = self._assess_cultural_awareness(
                response, query, category
            )
            
            # 5. Practical Utility Assessment
            dimension_scores[QualityDimension.PRACTICAL_UTILITY] = self._assess_practical_utility(
                response, query, category
            )
            
            # 6. Language Quality Assessment
            dimension_scores[QualityDimension.LANGUAGE_QUALITY] = self._assess_language_quality(
                response, query
            )
            
            # 7. Source Reliability Assessment
            dimension_scores[QualityDimension.SOURCE_RELIABILITY] = self._assess_source_reliability(
                response, category
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(dimension_scores)
            
            # Determine letter grade
            letter_grade = self._calculate_letter_grade(overall_score)
            
            # Identify strengths and weaknesses
            strengths, weaknesses = self._identify_strengths_weaknesses(dimension_scores)
            
            # Prioritize improvements
            improvement_priority = self._prioritize_improvements(dimension_scores, category)
            
            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(dimension_scores, response)
            
            return ComprehensiveQualityAssessment(
                overall_score=overall_score,
                letter_grade=letter_grade,
                dimension_scores=dimension_scores,
                strengths=strengths,
                weaknesses=weaknesses,
                improvement_priority=improvement_priority,
                confidence_level=confidence_level,
                assessment_timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            return self._get_fallback_assessment()
    
    def _initialize_scoring_weights(self) -> Dict[QualityDimension, float]:
        """Initialize scoring weights for different dimensions"""
        
        return {
            QualityDimension.ACCURACY: 0.25,           # Factual correctness
            QualityDimension.RELEVANCE: 0.20,          # Query relevance
            QualityDimension.COMPLETENESS: 0.15,       # Information completeness
            QualityDimension.CULTURAL_AWARENESS: 0.15, # Cultural sensitivity
            QualityDimension.PRACTICAL_UTILITY: 0.15,  # Actionable information
            QualityDimension.LANGUAGE_QUALITY: 0.05,   # Language and clarity
            QualityDimension.SOURCE_RELIABILITY: 0.05  # Source credibility
        }
    
    def _initialize_quality_benchmarks(self) -> Dict[str, Dict]:
        """Initialize quality benchmarks for different categories"""
        
        return {
            "museum": {
                "required_elements": ["location", "opening hours", "highlights", "how to get"],
                "quality_indicators": ["historical context", "practical tips", "cultural significance"],
                "excellence_markers": ["specific artifacts", "architectural details", "visitor guidance"]
            },
            "transportation": {
                "required_elements": ["route", "schedule", "cost", "stations"],
                "quality_indicators": ["alternative options", "real-time info", "accessibility"],
                "excellence_markers": ["step-by-step directions", "time estimates", "local tips"]
            },
            "restaurant": {
                "required_elements": ["location", "cuisine type", "price range"],
                "quality_indicators": ["ambiance", "specialties", "booking info"],
                "excellence_markers": ["specific dishes", "chef information", "dining experience"]
            },
            "cultural": {
                "required_elements": ["cultural context", "appropriate behavior", "significance"],
                "quality_indicators": ["historical background", "modern relevance", "respectful guidance"],
                "excellence_markers": ["deep cultural insights", "etiquette details", "personal experiences"]
            }
        }
    
    def _initialize_cultural_markers(self) -> Dict[str, List[str]]:
        """Initialize cultural sensitivity markers"""
        
        return {
            "positive_markers": [
                "respect", "appropriate", "cultural", "tradition", "etiquette",
                "modest", "prayer time", "ramadan", "halal", "local custom",
                "sensitivity", "awareness", "understanding"
            ],
            "negative_markers": [
                "ignore", "doesn't matter", "not important", "just do",
                "outdated", "primitive", "backward", "weird", "strange"
            ],
            "religious_sensitivity": [
                "mosque etiquette", "prayer times", "dress code", "remove shoes",
                "covering", "respect", "sacred", "holy"
            ],
            "social_awareness": [
                "tipping culture", "bargaining", "hospitality", "tea culture",
                "greeting customs", "family values", "social norms"
            ]
        }
    
    def _assess_accuracy(self, response: str, query: str, category: str, 
                        accuracy_result: Optional[Any]) -> QualityScore:
        """Assess accuracy quality dimension"""
        
        if accuracy_result:
            # Use provided accuracy result
            base_score = accuracy_result.accuracy_score * 5.0
            contributing_factors = [
                f"Historical accuracy: {accuracy_result.historical_accuracy:.2f}",
                f"Cultural accuracy: {accuracy_result.cultural_accuracy:.2f}",
                f"Real-time validity: {accuracy_result.real_time_validity:.2f}",
                f"Verified facts: {len(accuracy_result.verified_facts)}",
                f"Questionable claims: {len(accuracy_result.questionable_claims)}"
            ]
            improvement_suggestions = accuracy_result.recommendations
        else:
            # Basic accuracy assessment
            base_score, contributing_factors, improvement_suggestions = self._basic_accuracy_assessment(
                response, category
            )
        
        # Determine explanation
        if base_score >= 4.5:
            explanation = "Excellent factual accuracy with verified information"
        elif base_score >= 4.0:
            explanation = "Good accuracy with reliable information"
        elif base_score >= 3.0:
            explanation = "Moderate accuracy with some unverified claims"
        elif base_score >= 2.0:
            explanation = "Low accuracy with multiple questionable facts"
        else:
            explanation = "Poor accuracy with significant factual issues"
        
        return QualityScore(
            dimension=QualityDimension.ACCURACY,
            score=base_score,
            explanation=explanation,
            contributing_factors=contributing_factors,
            improvement_suggestions=improvement_suggestions
        )
    
    def _basic_accuracy_assessment(self, response: str, category: str) -> Tuple[float, List[str], List[str]]:
        """Basic accuracy assessment when detailed analysis unavailable"""
        
        response_lower = response.lower()
        score = 3.5  # Start with neutral score
        factors = []
        suggestions = []
        
        # Check for accuracy indicators
        accuracy_boosts = [
            ("specific numbers/dates", r"\b\d{4}\b|\b\d+\s*(?:meters?|km|minutes?)\b"),
            ("official sources mentioned", r"official|government|museum website"),
            ("proper nouns correctly used", r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*"),
            ("no vague language overuse", r"(?:might|could|perhaps|possibly).*(?:might|could|perhaps|possibly)")
        ]
        
        for desc, pattern in accuracy_boosts:
            if re.search(pattern, response):
                score += 0.3
                factors.append(f"✅ {desc}")
            elif "vague language" in desc and re.search(pattern, response):
                score -= 0.4
                factors.append(f"❌ Excessive vague language detected")
        
        # Check for accuracy red flags
        red_flags = [
            ("specific prices mentioned", r"\d+\s*(?:TL|lira|euro|dollar)"),
            ("absolute time claims", r"exactly\s+\d+\s*minutes?"),
            ("superlative claims", r"best|most\s+(?:popular|famous|visited)")
        ]
        
        for desc, pattern in red_flags:
            if re.search(pattern, response_lower):
                score -= 0.3
                factors.append(f"⚠️ {desc} - may become outdated")
                suggestions.append(f"Avoid {desc.lower()} without verification")
        
        score = max(0.0, min(5.0, score))
        
        if not suggestions:
            suggestions = ["Verify specific claims with official sources"]
        
        return score, factors, suggestions
    
    def _assess_relevance(self, response: str, query: str, category: str) -> QualityScore:
        """Assess relevance quality dimension"""
        
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Extract key terms from query
        query_words = set(word.lower() for word in re.findall(r'\b\w+\b', query) if len(word) > 3)
        response_words = set(word.lower() for word in re.findall(r'\b\w+\b', response) if len(word) > 3)
        
        # Calculate word overlap
        overlap = len(query_words.intersection(response_words))
        relevance_ratio = overlap / max(1, len(query_words))
        
        base_score = relevance_ratio * 3.0  # Start with word overlap score
        
        contributing_factors = [f"Query-response word overlap: {relevance_ratio:.2f}"]
        
        # Category-specific relevance checks
        category_keywords = {
            "museum": ["museum", "exhibition", "artifact", "visit", "gallery"],
            "transportation": ["transport", "metro", "bus", "tram", "ferry", "route"],
            "restaurant": ["restaurant", "food", "cuisine", "eat", "dining"],
            "cultural": ["culture", "tradition", "custom", "etiquette", "respect"]
        }
        
        if category in category_keywords:
            category_matches = sum(1 for keyword in category_keywords[category] 
                                 if keyword in response_lower)
            category_score = min(2.0, category_matches * 0.4)
            base_score += category_score
            contributing_factors.append(f"Category relevance: {category_score:.1f}/2.0")
        
        # Direct question answering
        question_indicators = ["what", "where", "when", "how", "why", "which"]
        if any(indicator in query_lower for indicator in question_indicators):
            if any(answer_word in response_lower for answer_word in ["located", "address", "time", "because", "since"]):
                base_score += 0.5
                contributing_factors.append("✅ Direct question answered")
            else:
                contributing_factors.append("⚠️ Direct question not clearly answered")
        
        score = min(5.0, base_score)
        
        # Generate explanation and suggestions
        if score >= 4.5:
            explanation = "Highly relevant response directly addressing query"
            suggestions = ["Maintain focused relevance"]
        elif score >= 3.5:
            explanation = "Good relevance with minor tangential content"
            suggestions = ["Focus more directly on specific query aspects"]
        elif score >= 2.5:
            explanation = "Moderate relevance but some off-topic content"
            suggestions = ["Remove tangential information", "Address query more directly"]
        else:
            explanation = "Low relevance - response doesn't adequately address query"
            suggestions = ["Completely refocus response on query", "Identify and address main question"]
        
        return QualityScore(
            dimension=QualityDimension.RELEVANCE,
            score=score,
            explanation=explanation,
            contributing_factors=contributing_factors,
            improvement_suggestions=suggestions
        )
    
    def _assess_completeness(self, response: str, query: str, category: str) -> QualityScore:
        """Assess completeness quality dimension"""
        
        response_lower = response.lower()
        
        # Get category-specific completeness requirements
        benchmarks = self.quality_benchmarks.get(category, self.quality_benchmarks["museum"])
        
        required_elements = benchmarks["required_elements"]
        quality_indicators = benchmarks["quality_indicators"]
        excellence_markers = benchmarks["excellence_markers"]
        
        # Check required elements
        required_coverage = 0
        covered_elements = []
        for element in required_elements:
            if any(keyword in response_lower for keyword in element.split()):
                required_coverage += 1
                covered_elements.append(element)
        
        required_score = (required_coverage / len(required_elements)) * 3.0
        
        # Check quality indicators
        quality_coverage = 0
        quality_elements = []
        for indicator in quality_indicators:
            if any(keyword in response_lower for keyword in indicator.split()):
                quality_coverage += 1
                quality_elements.append(indicator)
        
        quality_score = (quality_coverage / len(quality_indicators)) * 1.5
        
        # Check excellence markers
        excellence_coverage = 0
        excellence_elements = []
        for marker in excellence_markers:
            if any(keyword in response_lower for keyword in marker.split()):
                excellence_coverage += 1
                excellence_elements.append(marker)
        
        excellence_score = (excellence_coverage / len(excellence_markers)) * 0.5
        
        total_score = required_score + quality_score + excellence_score
        
        contributing_factors = [
            f"Required elements covered: {required_coverage}/{len(required_elements)} ({covered_elements})",
            f"Quality indicators present: {quality_coverage}/{len(quality_indicators)} ({quality_elements})",
            f"Excellence markers: {excellence_coverage}/{len(excellence_markers)} ({excellence_elements})"
        ]
        
        # Generate suggestions
        suggestions = []
        missing_required = [elem for elem in required_elements if elem not in covered_elements]
        if missing_required:
            suggestions.append(f"Add missing required elements: {', '.join(missing_required)}")
        
        missing_quality = [ind for ind in quality_indicators if ind not in quality_elements]
        if missing_quality and len(missing_quality) <= 2:
            suggestions.append(f"Enhance with: {', '.join(missing_quality)}")
        
        if excellence_coverage == 0:
            suggestions.append("Add specific details or expert insights for excellence")
        
        # Explanation
        if total_score >= 4.5:
            explanation = "Comprehensive response covering all essential aspects"
        elif total_score >= 3.5:
            explanation = "Good completeness with most important elements covered"
        elif total_score >= 2.5:
            explanation = "Moderate completeness - missing some important elements"
        else:
            explanation = "Incomplete response - significant gaps in information"
        
        return QualityScore(
            dimension=QualityDimension.COMPLETENESS,
            score=total_score,
            explanation=explanation,
            contributing_factors=contributing_factors,
            improvement_suggestions=suggestions if suggestions else ["Response is comprehensive"]
        )
    
    def _assess_cultural_awareness(self, response: str, query: str, category: str) -> QualityScore:
        """Assess cultural awareness quality dimension"""
        
        response_lower = response.lower()
        score = 2.5  # Start with neutral score
        contributing_factors = []
        
        # Check for positive cultural markers
        positive_count = 0
        for marker in self.cultural_sensitivity_markers["positive_markers"]:
            if marker in response_lower:
                positive_count += 1
                score += 0.2
        
        if positive_count > 0:
            contributing_factors.append(f"✅ Positive cultural awareness: {positive_count} indicators")
        
        # Check for negative cultural markers
        negative_count = 0
        for marker in self.cultural_sensitivity_markers["negative_markers"]:
            if marker in response_lower:
                negative_count += 1
                score -= 0.5
        
        if negative_count > 0:
            contributing_factors.append(f"❌ Cultural insensitivity: {negative_count} negative markers")
        
        # Religious sensitivity check
        religious_mentions = 0
        for marker in self.cultural_sensitivity_markers["religious_sensitivity"]:
            if marker in response_lower:
                religious_mentions += 1
                score += 0.3
        
        if religious_mentions > 0:
            contributing_factors.append(f"✅ Religious sensitivity: {religious_mentions} appropriate mentions")
        
        # Social awareness check
        social_mentions = 0
        for marker in self.cultural_sensitivity_markers["social_awareness"]:
            if marker in response_lower:
                social_mentions += 1
                score += 0.2
        
        if social_mentions > 0:
            contributing_factors.append(f"✅ Social awareness: {social_mentions} cultural insights")
        
        # Turkish language integration (bonus)
        turkish_terms = ["merhaba", "teşekkür", "affedersiniz", "kahvaltı", "çay", "namaz", "ramazan"]
        turkish_count = sum(1 for term in turkish_terms if term in response_lower)
        if turkish_count > 0:
            score += 0.1 * turkish_count
            contributing_factors.append(f"✅ Turkish language integration: {turkish_count} terms")
        
        score = max(0.0, min(5.0, score))
        
        # Generate explanation and suggestions
        if score >= 4.5:
            explanation = "Excellent cultural awareness and sensitivity"
            suggestions = ["Maintain high cultural sensitivity standards"]
        elif score >= 3.5:
            explanation = "Good cultural awareness with appropriate sensitivity"
            suggestions = ["Continue demonstrating cultural understanding"]
        elif score >= 2.5:
            explanation = "Moderate cultural awareness - room for improvement"
            suggestions = ["Include more cultural context", "Show greater sensitivity to local customs"]
        else:
            explanation = "Low cultural awareness - significant improvement needed"
            suggestions = ["Research Turkish cultural norms", "Avoid insensitive language", "Include respectful cultural guidance"]
        
        return QualityScore(
            dimension=QualityDimension.CULTURAL_AWARENESS,
            score=score,
            explanation=explanation,
            contributing_factors=contributing_factors,
            improvement_suggestions=suggestions
        )
    
    def _assess_practical_utility(self, response: str, query: str, category: str) -> QualityScore:
        """Assess practical utility quality dimension"""
        
        response_lower = response.lower()
        score = 0.0
        contributing_factors = []
        
        # Actionable information indicators
        actionable_patterns = [
            (r"how to get|transport|metro|bus", "Transportation guidance", 1.0),
            (r"opening hours?|open.*\d|closed.*\d", "Operating hours", 0.8),
            (r"address|located at|situated", "Location information", 0.8),
            (r"recommend|suggest|should visit", "Recommendations", 0.6),
            (r"cost|price|fee|ticket", "Cost information", 0.7),
            (r"book|reserve|contact", "Booking information", 0.5),
            (r"avoid|careful|warning|caution", "Safety/warning advice", 0.6),
            (r"best time|when to visit", "Timing advice", 0.5)
        ]
        
        for pattern, desc, weight in actionable_patterns:
            if re.search(pattern, response_lower):
                score += weight
                contributing_factors.append(f"✅ {desc}")
        
        # Specific vs vague information
        specific_patterns = [
            (r"\b\d{1,2}:\d{2}\b", "Specific times"),
            (r"\b\d+\s*(?:meters?|km|minutes?)\b", "Specific measurements"),
            (r"line\s+[A-Z]?\d+|M\d+|T\d+", "Specific transport lines"),
            (r"station\s+\w+|stop\s+\w+", "Specific stations/stops")
        ]
        
        specificity_count = 0
        for pattern, desc in specific_patterns:
            if re.search(pattern, response):
                specificity_count += 1
                score += 0.2
        
        if specificity_count > 0:
            contributing_factors.append(f"✅ Specific details: {specificity_count} instances")
        
        # Deduct for vague language
        vague_patterns = ["might", "could", "perhaps", "possibly", "maybe"]
        vague_count = sum(1 for term in vague_patterns if term in response_lower)
        if vague_count > 3:
            score -= 0.5
            contributing_factors.append(f"❌ Excessive vague language: {vague_count} instances")
        
        score = max(0.0, min(5.0, score))
        
        # Generate explanation and suggestions
        if score >= 4.0:
            explanation = "Highly practical with actionable information"
            suggestions = ["Maintain practical focus"]
        elif score >= 3.0:
            explanation = "Good practical utility with useful information"
            suggestions = ["Add more specific details where possible"]
        elif score >= 2.0:
            explanation = "Moderate practical utility - needs more actionable content"
            suggestions = ["Include more how-to information", "Add specific times and locations"]
        else:
            explanation = "Low practical utility - mostly general information"
            suggestions = ["Focus on actionable advice", "Include specific practical details", "Provide step-by-step guidance"]
        
        return QualityScore(
            dimension=QualityDimension.PRACTICAL_UTILITY,
            score=score,
            explanation=explanation,
            contributing_factors=contributing_factors,
            improvement_suggestions=suggestions
        )
    
    def _assess_language_quality(self, response: str, query: str) -> QualityScore:
        """Assess language quality and clarity"""
        
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if not sentences:
            return QualityScore(
                dimension=QualityDimension.LANGUAGE_QUALITY,
                score=0.0,
                explanation="No valid sentences found",
                contributing_factors=["❌ Empty or invalid response"],
                improvement_suggestions=["Provide a proper response"]
            )
        
        score = 3.0  # Start with neutral score
        contributing_factors = []
        
        # Sentence length analysis
        word_counts = [len(s.split()) for s in sentences]
        avg_sentence_length = sum(word_counts) / len(word_counts)
        
        if 10 <= avg_sentence_length <= 25:
            score += 0.5
            contributing_factors.append("✅ Good sentence length balance")
        elif avg_sentence_length > 30:
            score -= 0.3
            contributing_factors.append("❌ Sentences too long (hard to read)")
        elif avg_sentence_length < 5:
            score -= 0.2
            contributing_factors.append("❌ Sentences too short (choppy)")
        
        # Grammar and structure indicators
        grammar_score = 0
        
        # Proper capitalization
        properly_capitalized = sum(1 for s in sentences if s and s[0].isupper())
        if properly_capitalized == len(sentences):
            grammar_score += 0.3
            contributing_factors.append("✅ Proper capitalization")
        
        # Variety in sentence starters
        starters = [s.split()[0].lower() for s in sentences if s.split()]
        unique_starters = len(set(starters))
        if unique_starters / max(1, len(starters)) > 0.7:
            grammar_score += 0.2
            contributing_factors.append("✅ Good sentence variety")
        
        # Transitional phrases
        transitions = ["however", "additionally", "furthermore", "meanwhile", "consequently", "therefore"]
        transition_count = sum(1 for trans in transitions if trans in response.lower())
        if transition_count > 0:
            grammar_score += 0.2
            contributing_factors.append(f"✅ Good flow: {transition_count} transitions")
        
        score += grammar_score
        
        # Check for common errors
        error_patterns = [
            (r"\s{2,}", "Multiple spaces"),
            (r"[.]{2,}", "Multiple periods"),
            (r"[A-Z]{3,}", "Excessive capitalization")
        ]
        
        error_count = 0
        for pattern, desc in error_patterns:
            if re.search(pattern, response):
                error_count += 1
                score -= 0.2
        
        if error_count > 0:
            contributing_factors.append(f"❌ Formatting errors: {error_count}")
        
        score = max(0.0, min(5.0, score))
        
        # Generate explanation and suggestions
        if score >= 4.5:
            explanation = "Excellent language quality and clarity"
            suggestions = ["Maintain high language standards"]
        elif score >= 3.5:
            explanation = "Good language quality with clear communication"
            suggestions = ["Minor language improvements possible"]
        elif score >= 2.5:
            explanation = "Adequate language quality - room for improvement"
            suggestions = ["Improve sentence structure", "Enhance clarity"]
        else:
            explanation = "Poor language quality affecting readability"
            suggestions = ["Improve grammar and structure", "Simplify complex sentences", "Fix formatting issues"]
        
        return QualityScore(
            dimension=QualityDimension.LANGUAGE_QUALITY,
            score=score,
            explanation=explanation,
            contributing_factors=contributing_factors,
            improvement_suggestions=suggestions
        )
    
    def _assess_source_reliability(self, response: str, category: str) -> QualityScore:
        """Assess source reliability indicators"""
        
        response_lower = response.lower()
        score = 2.5  # Neutral start
        contributing_factors = []
        
        # Check for source mentions
        reliable_source_indicators = [
            ("official", 0.5),
            ("government", 0.5),
            ("museum website", 0.4),
            ("verified", 0.3),
            ("according to", 0.3),
            ("documented", 0.3)
        ]
        
        source_score = 0
        for indicator, weight in reliable_source_indicators:
            if indicator in response_lower:
                source_score += weight
                contributing_factors.append(f"✅ {indicator.title()} source mentioned")
        
        score += min(2.0, source_score)
        
        # Penalty for unreliable indicators
        unreliable_indicators = [
            ("i heard", -0.5),
            ("someone told me", -0.5),
            ("i think", -0.3),
            ("probably", -0.2),
            ("maybe", -0.2)
        ]
        
        for indicator, penalty in unreliable_indicators:
            if indicator in response_lower:
                score += penalty
                contributing_factors.append(f"❌ Unreliable language: '{indicator}'")
        
        # Check for appropriate disclaimers
        disclaimers = [
            "check current",
            "verify before",
            "subject to change",
            "please confirm"
        ]
        
        disclaimer_count = sum(1 for disc in disclaimers if disc in response_lower)
        if disclaimer_count > 0:
            score += 0.3
            contributing_factors.append(f"✅ Appropriate disclaimers: {disclaimer_count}")
        
        score = max(0.0, min(5.0, score))
        
        # Generate explanation and suggestions
        if score >= 4.0:
            explanation = "Good source reliability with verification indicators"
            suggestions = ["Continue referencing reliable sources"]
        elif score >= 3.0:
            explanation = "Moderate source reliability"
            suggestions = ["Include more source references", "Add verification disclaimers"]
        else:
            explanation = "Low source reliability indicators"
            suggestions = ["Reference official sources", "Avoid unverified claims", "Add appropriate disclaimers"]
        
        return QualityScore(
            dimension=QualityDimension.SOURCE_RELIABILITY,
            score=score,
            explanation=explanation,
            contributing_factors=contributing_factors,
            improvement_suggestions=suggestions
        )
    
    def _calculate_overall_score(self, dimension_scores: Dict[QualityDimension, QualityScore]) -> float:
        """Calculate weighted overall quality score"""
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for dimension, quality_score in dimension_scores.items():
            weight = self.scoring_weights[dimension]
            total_weighted_score += quality_score.score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_letter_grade(self, overall_score: float) -> str:
        """Convert numerical score to letter grade"""
        
        if overall_score >= 4.8:
            return "A+"
        elif overall_score >= 4.5:
            return "A"
        elif overall_score >= 4.0:
            return "A-"
        elif overall_score >= 3.7:
            return "B+"
        elif overall_score >= 3.3:
            return "B"
        elif overall_score >= 3.0:
            return "B-"
        elif overall_score >= 2.7:
            return "C+"
        elif overall_score >= 2.3:
            return "C"
        elif overall_score >= 2.0:
            return "C-"
        elif overall_score >= 1.7:
            return "D+"
        elif overall_score >= 1.3:
            return "D"
        elif overall_score >= 1.0:
            return "D-"
        else:
            return "F"
    
    def _identify_strengths_weaknesses(self, dimension_scores: Dict[QualityDimension, QualityScore]) -> Tuple[List[str], List[str]]:
        """Identify response strengths and weaknesses"""
        
        strengths = []
        weaknesses = []
        
        for dimension, quality_score in dimension_scores.items():
            if quality_score.score >= 4.0:
                strengths.append(f"Strong {dimension.value}: {quality_score.explanation}")
            elif quality_score.score <= 2.5:
                weaknesses.append(f"Weak {dimension.value}: {quality_score.explanation}")
        
        return strengths, weaknesses
    
    def _prioritize_improvements(self, dimension_scores: Dict[QualityDimension, QualityScore], category: str) -> List[str]:
        """Prioritize improvement areas based on scores and weights"""
        
        # Calculate improvement impact (weight * (5.0 - current_score))
        improvement_impact = []
        
        for dimension, quality_score in dimension_scores.items():
            weight = self.scoring_weights[dimension]
            potential_gain = (5.0 - quality_score.score) * weight
            
            if potential_gain > 0.2:  # Only consider meaningful improvements
                improvement_impact.append((potential_gain, dimension, quality_score.improvement_suggestions[0] if quality_score.improvement_suggestions else "General improvement needed"))
        
        # Sort by impact (highest first)
        improvement_impact.sort(reverse=True)
        
        return [f"{dimension.value.title()}: {suggestion}" for _, dimension, suggestion in improvement_impact[:5]]
    
    def _calculate_confidence_level(self, dimension_scores: Dict[QualityDimension, QualityScore], response: str) -> float:
        """Calculate confidence level in the assessment"""
        
        # Base confidence on response length and detail
        word_count = len(response.split())
        
        if word_count < 20:
            base_confidence = 0.6  # Low confidence for very short responses
        elif word_count < 100:
            base_confidence = 0.8  # Medium confidence
        else:
            base_confidence = 0.9  # High confidence for detailed responses
        
        # Adjust based on score consistency
        scores = [qs.score for qs in dimension_scores.values()]
        score_variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
        
        if score_variance > 2.0:  # High variance means inconsistent quality
            base_confidence -= 0.1
        
        return max(0.0, min(1.0, base_confidence))
    
    def _get_fallback_assessment(self) -> ComprehensiveQualityAssessment:
        """Fallback assessment when analysis fails"""
        
        fallback_scores = {}
        for dimension in QualityDimension:
            fallback_scores[dimension] = QualityScore(
                dimension=dimension,
                score=2.5,
                explanation="Assessment unavailable",
                contributing_factors=["Analysis failed"],
                improvement_suggestions=["Manual review needed"]
            )
        
        return ComprehensiveQualityAssessment(
            overall_score=2.5,
            letter_grade="C",
            dimension_scores=fallback_scores,
            strengths=[],
            weaknesses=["Assessment system unavailable"],
            improvement_priority=["Manual quality review recommended"],
            confidence_level=0.3,
            assessment_timestamp=datetime.now().isoformat()
        )

# Global instance
enhanced_quality_scorer = EnhancedQualityScorer()

def assess_response_quality(response: str, query: str, category: str = "general", 
                           accuracy_result: Optional[Any] = None) -> ComprehensiveQualityAssessment:
    """
    Main function to assess response quality
    
    Args:
        response: AI response to assess
        query: Original user query
        category: Query category
        accuracy_result: Optional accuracy analysis result
        
    Returns:
        ComprehensiveQualityAssessment with detailed quality scoring
    """
    return enhanced_quality_scorer.assess_response_quality(response, query, category, accuracy_result)

def get_quality_score_only(response: str, query: str, category: str = "general") -> float:
    """Get just the overall quality score for quick evaluation"""
    assessment = enhanced_quality_scorer.assess_response_quality(response, query, category)
    return assessment.overall_score

def get_improvement_suggestions(response: str, query: str, category: str = "general") -> List[str]:
    """Get prioritized improvement suggestions"""
    assessment = enhanced_quality_scorer.assess_response_quality(response, query, category)
    return assessment.improvement_priority
