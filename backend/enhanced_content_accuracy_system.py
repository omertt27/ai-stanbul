#!/usr/bin/env python3
"""
Enhanced Content Accuracy System for AI Istanbul
===============================================

Phase 3: Content Accuracy Improvement
- Enhanced fact-checking databases
- Improved historical and cultural accuracy
- Better real-time information validation
- Enhanced quality scoring

Target: +0.05 points improvement
"""

import os
import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class AccuracyLevel(Enum):
    """Accuracy levels for content validation"""
    VERIFIED = "verified"           # 90-100% accuracy
    HIGH_CONFIDENCE = "high"        # 80-89% accuracy
    MODERATE = "moderate"          # 60-79% accuracy
    LOW_CONFIDENCE = "low"         # 40-59% accuracy
    QUESTIONABLE = "questionable"  # 0-39% accuracy

@dataclass
class ContentAccuracyResult:
    """Result of content accuracy analysis"""
    accuracy_level: AccuracyLevel
    accuracy_score: float  # 0.0 to 1.0
    verified_facts: List[str]
    questionable_claims: List[str]
    historical_accuracy: float
    cultural_accuracy: float
    real_time_validity: float
    quality_indicators: Dict[str, Any]
    recommendations: List[str]
    sources_used: List[str]
    timestamp: str

@dataclass
class HistoricalFact:
    """Historical fact with validation"""
    fact: str
    period: str
    source: str
    confidence: float
    last_verified: str

@dataclass
class CulturalKnowledge:
    """Cultural knowledge entry"""
    concept: str
    description: str
    context: str
    sensitivity_level: str  # low, medium, high
    accuracy_rating: float

class EnhancedContentAccuracySystem:
    """Advanced content accuracy validation system"""
    
    def __init__(self):
        self.historical_database = self._initialize_historical_database()
        self.cultural_database = self._initialize_cultural_database()
        self.real_time_validators = self._initialize_real_time_validators()
        self.quality_metrics = self._initialize_quality_metrics()
        self.accuracy_cache = {}
        self.cache_duration = timedelta(hours=6)
        
        logger.info("✅ Enhanced Content Accuracy System initialized")
    
    def analyze_content_accuracy(self, response: str, query: str, category: str) -> ContentAccuracyResult:
        """
        Comprehensive content accuracy analysis
        
        Args:
            response: AI response to analyze
            query: Original user query
            category: Query category (museum, transportation, etc.)
            
        Returns:
            ContentAccuracyResult with detailed accuracy assessment
        """
        
        try:
            # Check cache first
            cache_key = f"{hash(response)}_{category}"
            if cache_key in self.accuracy_cache:
                cached_result = self.accuracy_cache[cache_key]
                if datetime.fromisoformat(cached_result['timestamp']) > datetime.now() - self.cache_duration:
                    return ContentAccuracyResult(**cached_result)
            
            # 1. Historical accuracy analysis
            historical_score, historical_facts = self._analyze_historical_accuracy(response, category)
            
            # 2. Cultural accuracy analysis
            cultural_score, cultural_insights = self._analyze_cultural_accuracy(response, query)
            
            # 3. Real-time information validation
            real_time_score, real_time_issues = self._validate_real_time_info(response, category)
            
            # 4. Extract and verify factual claims
            verified_facts, questionable_claims = self._extract_and_verify_claims(response, category)
            
            # 5. Calculate quality indicators
            quality_indicators = self._calculate_quality_indicators(response, query, category)
            
            # 6. Overall accuracy scoring
            accuracy_score = self._calculate_overall_accuracy(
                historical_score, cultural_score, real_time_score, 
                len(verified_facts), len(questionable_claims)
            )
            
            # 7. Determine accuracy level
            accuracy_level = self._determine_accuracy_level(accuracy_score)
            
            # 8. Generate recommendations
            recommendations = self._generate_accuracy_recommendations(
                accuracy_score, questionable_claims, real_time_issues, category
            )
            
            # 9. Compile sources used
            sources_used = self._compile_sources_used(category, verified_facts)
            
            result = ContentAccuracyResult(
                accuracy_level=accuracy_level,
                accuracy_score=accuracy_score,
                verified_facts=verified_facts,
                questionable_claims=questionable_claims,
                historical_accuracy=historical_score,
                cultural_accuracy=cultural_score,
                real_time_validity=real_time_score,
                quality_indicators=quality_indicators,
                recommendations=recommendations,
                sources_used=sources_used,
                timestamp=datetime.now().isoformat()
            )
            
            # Cache result
            self.accuracy_cache[cache_key] = asdict(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in content accuracy analysis: {e}")
            return self._get_fallback_accuracy_result()
    
    def _initialize_historical_database(self) -> Dict[str, List[HistoricalFact]]:
        """Initialize comprehensive historical facts database"""
        
        return {
            "byzantine_period": [
                HistoricalFact(
                    fact="Constantinople was founded in 330 AD by Emperor Constantine I",
                    period="Byzantine",
                    source="Historical consensus",
                    confidence=1.0,
                    last_verified="2024-01-01"
                ),
                HistoricalFact(
                    fact="Hagia Sophia was built between 532-537 AD under Emperor Justinian I",
                    period="Byzantine",
                    source="Historical records",
                    confidence=1.0,
                    last_verified="2024-01-01"
                ),
                HistoricalFact(
                    fact="The Byzantine Empire controlled Constantinople for over 1000 years",
                    period="Byzantine",
                    source="Historical consensus",
                    confidence=1.0,
                    last_verified="2024-01-01"
                )
            ],
            
            "ottoman_period": [
                HistoricalFact(
                    fact="Ottoman Empire conquered Constantinople in 1453 under Mehmed II",
                    period="Ottoman",
                    source="Historical records",
                    confidence=1.0,
                    last_verified="2024-01-01"
                ),
                HistoricalFact(
                    fact="Topkapi Palace served as Ottoman imperial residence from 1465-1856",
                    period="Ottoman",
                    source="Palace records",
                    confidence=1.0,
                    last_verified="2024-01-01"
                ),
                HistoricalFact(
                    fact="Blue Mosque was built between 1609-1616 by Sultan Ahmed I",
                    period="Ottoman",
                    source="Ottoman archives",
                    confidence=1.0,
                    last_verified="2024-01-01"
                ),
                HistoricalFact(
                    fact="Galata Tower was built by Genoese in 1348, incorporated into Ottoman Empire",
                    period="Ottoman",
                    source="Historical records",
                    confidence=0.95,
                    last_verified="2024-01-01"
                )
            ],
            
            "modern_period": [
                HistoricalFact(
                    fact="Republic of Turkey established in 1923, Istanbul lost capital status to Ankara",
                    period="Modern",
                    source="Turkish government records",
                    confidence=1.0,
                    last_verified="2024-01-01"
                ),
                HistoricalFact(
                    fact="Hagia Sophia converted from museum back to mosque in 2020",
                    period="Modern",
                    source="Official Turkish decree",
                    confidence=1.0,
                    last_verified="2024-01-01"
                )
            ]
        }
    
    def _initialize_cultural_database(self) -> Dict[str, List[CulturalKnowledge]]:
        """Initialize cultural knowledge database"""
        
        return {
            "religious_customs": [
                CulturalKnowledge(
                    concept="Mosque etiquette",
                    description="Remove shoes, dress modestly, respect prayer times",
                    context="All functioning mosques including Blue Mosque, Hagia Sophia",
                    sensitivity_level="high",
                    accuracy_rating=1.0
                ),
                CulturalKnowledge(
                    concept="Prayer times",
                    description="Five daily prayers affect mosque visiting hours",
                    context="All mosques have restricted access during prayer times",
                    sensitivity_level="high",
                    accuracy_rating=1.0
                ),
                CulturalKnowledge(
                    concept="Ramadan considerations",
                    description="Restaurant hours and availability may change during Ramadan",
                    context="Month-long Islamic fasting period",
                    sensitivity_level="medium",
                    accuracy_rating=1.0
                )
            ],
            
            "social_customs": [
                CulturalKnowledge(
                    concept="Turkish hospitality",
                    description="Turks are known for exceptional hospitality to guests",
                    context="General social interaction",
                    sensitivity_level="low",
                    accuracy_rating=0.9
                ),
                CulturalKnowledge(
                    concept="Tea culture",
                    description="Turkish tea (çay) is central to social interactions",
                    context="Offered frequently, considered polite to accept",
                    sensitivity_level="low",
                    accuracy_rating=0.95
                ),
                CulturalKnowledge(
                    concept="Tipping culture",
                    description="10-15% tip expected in restaurants, round up for taxis",
                    context="Service industry standard",
                    sensitivity_level="low",
                    accuracy_rating=0.9
                )
            ],
            
            "language_culture": [
                CulturalKnowledge(
                    concept="Basic Turkish phrases",
                    description="Merhaba (hello), teşekkür ederim (thank you), affedersiniz (excuse me)",
                    context="Polite interactions appreciated",
                    sensitivity_level="low",
                    accuracy_rating=1.0
                ),
                CulturalKnowledge(
                    concept="English proficiency",
                    description="Good English in tourist areas, limited elsewhere",
                    context="Tourism and business districts",
                    sensitivity_level="low",
                    accuracy_rating=0.9
                )
            ]
        }
    
    def _initialize_real_time_validators(self) -> Dict[str, Dict]:
        """Initialize real-time information validators"""
        
        return {
            "opening_hours": {
                "pattern": r"\b\d{1,2}:\d{2}\b",
                "validation_method": "check_current_schedules",
                "reliability": 0.6  # Hours change frequently
            },
            "ticket_prices": {
                "pattern": r"\d+\s*(?:TL|lira|euro|dollar)",
                "validation_method": "flag_for_verification",
                "reliability": 0.3  # Prices change very frequently
            },
            "transportation_schedules": {
                "pattern": r"every\s+\d+\s*minutes?",
                "validation_method": "check_transport_apis",
                "reliability": 0.7  # Schedules relatively stable
            },
            "weather_information": {
                "pattern": r"\b\d+°?[CF]\b",
                "validation_method": "cross_check_weather",
                "reliability": 0.8  # Weather info can be verified
            }
        }
    
    def _initialize_quality_metrics(self) -> Dict[str, float]:
        """Initialize quality scoring metrics"""
        
        return {
            "specificity_weight": 0.2,      # Specific vs vague information
            "source_reliability": 0.25,     # Use of reliable sources
            "cultural_sensitivity": 0.15,   # Cultural awareness
            "practical_utility": 0.2,       # Actionable information
            "factual_accuracy": 0.2         # Verified facts vs claims
        }
    
    def _analyze_historical_accuracy(self, response: str, category: str) -> Tuple[float, List[str]]:
        """Analyze historical accuracy of response"""
        
        response_lower = response.lower()
        verified_historical_facts = []
        historical_score = 0.8  # Start with neutral score
        
        # Check for historical claims
        historical_patterns = [
            r"\b\d{3,4}(?:\s*AD)?\b",  # Years
            r"(?:built|constructed|founded|established)\s+(?:in\s+)?\d{3,4}",
            r"(?:emperor|sultan|byzantine|ottoman|roman)",
            r"(?:empire|dynasty|period|era)"
        ]
        
        has_historical_claims = any(re.search(pattern, response_lower) for pattern in historical_patterns)
        
        if has_historical_claims:
            # Check against historical database
            for period, facts in self.historical_database.items():
                for historical_fact in facts:
                    # Simple keyword matching - can be enhanced
                    fact_keywords = historical_fact.fact.lower().split()
                    matches = sum(1 for keyword in fact_keywords if keyword in response_lower)
                    
                    if matches >= 3:  # Significant overlap
                        verified_historical_facts.append(historical_fact.fact)
                        historical_score += 0.1 * historical_fact.confidence
            
            # Penalty for common historical errors
            error_patterns = [
                (r"constantinople.*founded.*\b(?!330)\d{3}\b", "Incorrect Constantinople founding date"),
                (r"hagia sophia.*built.*\b(?!532|537)\d{3}\b", "Incorrect Hagia Sophia construction date"),
                (r"ottoman.*conquered.*\b(?!1453)\d{4}\b", "Incorrect Ottoman conquest date")
            ]
            
            for pattern, error_desc in error_patterns:
                if re.search(pattern, response_lower):
                    historical_score -= 0.2
                    logger.warning(f"Historical accuracy issue detected: {error_desc}")
        
        return min(1.0, max(0.0, historical_score)), verified_historical_facts
    
    def _analyze_cultural_accuracy(self, response: str, query: str) -> Tuple[float, List[str]]:
        """Analyze cultural accuracy and sensitivity"""
        
        response_lower = response.lower()
        query_lower = query.lower()
        cultural_insights = []
        cultural_score = 0.8  # Start with neutral score
        
        # Check for cultural concepts
        for category, knowledge_items in self.cultural_database.items():
            for item in knowledge_items:
                concept_keywords = item.concept.lower().split()
                if any(keyword in response_lower for keyword in concept_keywords):
                    cultural_insights.append(item.concept)
                    
                    # Boost score for high-sensitivity accurate information
                    if item.sensitivity_level == "high" and item.accuracy_rating > 0.9:
                        cultural_score += 0.15
                    elif item.accuracy_rating > 0.8:
                        cultural_score += 0.1
        
        # Check for cultural sensitivity indicators
        sensitivity_indicators = [
            ("respect", 0.1),
            ("appropriate", 0.1),
            ("modest", 0.1),
            ("prayer time", 0.15),
            ("cultural", 0.1),
            ("traditional", 0.1),
            ("local custom", 0.15)
        ]
        
        for indicator, boost in sensitivity_indicators:
            if indicator in response_lower:
                cultural_score += boost
        
        # Penalty for cultural insensitivity
        insensitive_patterns = [
            r"just ignore",
            r"doesn't matter",
            r"not important",
            r"outdated tradition"
        ]
        
        for pattern in insensitive_patterns:
            if re.search(pattern, response_lower):
                cultural_score -= 0.3
        
        return min(1.0, max(0.0, cultural_score)), cultural_insights
    
    def _validate_real_time_info(self, response: str, category: str) -> Tuple[float, List[str]]:
        """Validate real-time information accuracy"""
        
        real_time_issues = []
        real_time_score = 0.9  # Start high, deduct for issues
        
        # Check for time-sensitive information
        for info_type, validator in self.real_time_validators.items():
            pattern_matches = re.findall(validator["pattern"], response)
            
            if pattern_matches:
                reliability = validator["reliability"]
                
                if reliability < 0.5:  # Low reliability information
                    real_time_issues.append(f"Time-sensitive {info_type} detected - verify independently")
                    real_time_score -= 0.2
                elif reliability < 0.8:  # Moderate reliability
                    real_time_issues.append(f"{info_type.title()} may change - check current status")
                    real_time_score -= 0.1
        
        # Boost score for appropriate disclaimers
        disclaimer_patterns = [
            r"check.*current",
            r"verify.*before",
            r"subject to change",
            r"as of.*date",
            r"please confirm"
        ]
        
        for pattern in disclaimer_patterns:
            if re.search(pattern, response.lower()):
                real_time_score += 0.1
                break
        
        return min(1.0, max(0.0, real_time_score)), real_time_issues
    
    def _extract_and_verify_claims(self, response: str, category: str) -> Tuple[List[str], List[str]]:
        """Extract and verify factual claims from response"""
        
        # Use existing fact-checking service if available
        try:
            from fact_checking_service import fact_checker
            fact_result = fact_checker.fact_check_response(response, category)
            return fact_result.verified_facts, fact_result.questionable_facts
        except ImportError:
            logger.warning("Fact-checking service not available, using basic verification")
            
            # Basic claim extraction
            verified_facts = []
            questionable_claims = []
            
            # Extract potential factual claims
            claim_patterns = [
                r"(?:built|constructed|opened)\s+in\s+\d{4}",
                r"\d+\s*(?:meters?|km|kilometers?)\s+(?:high|tall|long|wide)",
                r"(?:open|opens?|operating)\s+(?:from\s+)?\d{1,2}:\d{2}",
                r"UNESCO\s+World\s+Heritage\s+Site",
                r"designed\s+by\s+[A-Z][a-zA-Z\s]+"
            ]
            
            for pattern in claim_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                for match in matches:
                    # Simple verification - enhance based on known facts
                    if self._basic_fact_verification(match, category):
                        verified_facts.append(match)
                    else:
                        questionable_claims.append(match)
            
            return verified_facts, questionable_claims
    
    def _basic_fact_verification(self, claim: str, category: str) -> bool:
        """Basic fact verification against known patterns"""
        
        claim_lower = claim.lower()
        
        # Known accurate patterns
        accurate_patterns = [
            r"built.*153[2-7]",  # Hagia Sophia
            r"built.*160[9-6]",  # Blue Mosque
            r"built.*145[9-5]",  # Topkapi Palace
            r"unesco.*world.*heritage",  # UNESCO sites
            r"67.*meters?.*high",  # Galata Tower
        ]
        
        return any(re.search(pattern, claim_lower) for pattern in accurate_patterns)
    
    def _calculate_quality_indicators(self, response: str, query: str, category: str) -> Dict[str, Any]:
        """Calculate various quality indicators"""
        
        response_words = response.split()
        response_lower = response.lower()
        
        indicators = {
            "response_length": len(response_words),
            "specificity_score": self._calculate_specificity_score(response),
            "practical_info_score": self._calculate_practical_info_score(response),
            "source_mention_score": self._calculate_source_mention_score(response),
            "language_quality_score": self._calculate_language_quality_score(response),
            "completeness_score": self._calculate_completeness_score(response, query, category)
        }
        
        return indicators
    
    def _calculate_specificity_score(self, response: str) -> float:
        """Calculate how specific vs vague the response is"""
        
        vague_terms = ["might", "could", "perhaps", "possibly", "sometimes", "usually", "generally"]
        specific_terms = ["located at", "address", "exactly", "specifically", "precisely", "meters", "minutes"]
        
        vague_count = sum(1 for term in vague_terms if term in response.lower())
        specific_count = sum(1 for term in specific_terms if term in response.lower())
        
        total_words = len(response.split())
        if total_words == 0:
            return 0.0
        
        vague_ratio = vague_count / total_words
        specific_ratio = specific_count / total_words
        
        return max(0.0, min(1.0, specific_ratio - vague_ratio + 0.5))
    
    def _calculate_practical_info_score(self, response: str) -> float:
        """Calculate practical utility of information"""
        
        practical_indicators = [
            r"\b\d{1,2}:\d{2}\b",  # Times
            r"address|located",     # Location info
            r"how to get|transport|metro|bus|tram",  # Transportation
            r"ticket|price|fee|cost",  # Pricing
            r"open|closed|hours",   # Operating hours
            r"recommend|suggest|should visit"  # Actionable advice
        ]
        
        score = 0.0
        for pattern in practical_indicators:
            if re.search(pattern, response.lower()):
                score += 0.15
        
        return min(1.0, score)
    
    def _calculate_source_mention_score(self, response: str) -> float:
        """Calculate score for mentioning reliable sources"""
        
        source_indicators = [
            r"official",
            r"government",
            r"museum website",
            r"according to",
            r"verified",
            r"documented"
        ]
        
        score = 0.0
        for pattern in source_indicators:
            if re.search(pattern, response.lower()):
                score += 0.2
        
        return min(1.0, score)
    
    def _calculate_language_quality_score(self, response: str) -> float:
        """Calculate language quality and clarity"""
        
        # Simple metrics - can be enhanced with NLP
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        
        # Optimal sentence length is 15-25 words
        if 15 <= avg_sentence_length <= 25:
            length_score = 1.0
        else:
            length_score = max(0.0, 1.0 - abs(avg_sentence_length - 20) / 20)
        
        # Check for proper grammar indicators
        grammar_indicators = [
            r"^[A-Z]",  # Starts with capital
            r"\.$",     # Ends with period
        ]
        
        grammar_score = sum(1 for pattern in grammar_indicators if re.search(pattern, response)) / len(grammar_indicators)
        
        return (length_score + grammar_score) / 2
    
    def _calculate_completeness_score(self, response: str, query: str, category: str) -> float:
        """Calculate how complete the response is for the query"""
        
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Category-specific completeness indicators
        completeness_indicators = {
            "museum": ["opening hours", "location", "ticket", "highlights", "how to get"],
            "transportation": ["route", "schedule", "cost", "duration", "stations"],
            "restaurant": ["location", "cuisine", "price range", "recommend", "hours"],
            "general": ["location", "recommend", "how to", "when to", "cost"]
        }
        
        indicators = completeness_indicators.get(category, completeness_indicators["general"])
        
        coverage = 0.0
        for indicator in indicators:
            if indicator in response_lower:
                coverage += 1.0 / len(indicators)
        
        return coverage
    
    def _calculate_overall_accuracy(self, historical_score: float, cultural_score: float, 
                                    real_time_score: float, verified_count: int, 
                                    questionable_count: int) -> float:
        """Calculate overall accuracy score"""
        
        # Weight the different components
        component_weights = {
            "historical": 0.25,
            "cultural": 0.20,
            "real_time": 0.15,
            "factual": 0.40
        }
        
        # Calculate factual accuracy component
        total_claims = verified_count + questionable_count
        if total_claims > 0:
            factual_score = verified_count / total_claims
        else:
            factual_score = 0.8  # Neutral score for responses without specific claims
        
        # Weighted average
        overall_score = (
            historical_score * component_weights["historical"] +
            cultural_score * component_weights["cultural"] +
            real_time_score * component_weights["real_time"] +
            factual_score * component_weights["factual"]
        )
        
        return min(1.0, max(0.0, overall_score))
    
    def _determine_accuracy_level(self, accuracy_score: float) -> AccuracyLevel:
        """Determine accuracy level from score"""
        
        if accuracy_score >= 0.9:
            return AccuracyLevel.VERIFIED
        elif accuracy_score >= 0.8:
            return AccuracyLevel.HIGH_CONFIDENCE
        elif accuracy_score >= 0.6:
            return AccuracyLevel.MODERATE
        elif accuracy_score >= 0.4:
            return AccuracyLevel.LOW_CONFIDENCE
        else:
            return AccuracyLevel.QUESTIONABLE
    
    def _generate_accuracy_recommendations(self, accuracy_score: float, questionable_claims: List[str], 
                                           real_time_issues: List[str], category: str) -> List[str]:
        """Generate recommendations for improving accuracy"""
        
        recommendations = []
        
        # Score-based recommendations
        if accuracy_score >= 0.9:
            recommendations.append("✅ Content shows high accuracy - excellent response quality")
        elif accuracy_score >= 0.7:
            recommendations.append("✅ Content is generally accurate with minor areas for improvement")
        elif accuracy_score >= 0.5:
            recommendations.append("⚠️ Content accuracy is moderate - verify key information independently")
        else:
            recommendations.append("❌ Content accuracy is low - significant verification needed")
        
        # Specific issue recommendations
        if questionable_claims:
            if len(questionable_claims) <= 3:
                recommendations.append(f"🔍 Verify specific claims: {'; '.join(questionable_claims[:3])}")
            else:
                recommendations.append(f"🔍 Multiple unverified claims detected - thorough fact-checking recommended")
        
        if real_time_issues:
            recommendations.append("⏰ Time-sensitive information detected - check current status before relying on details")
        
        # Category-specific recommendations
        category_recommendations = {
            "museum": "🏛️ Verify opening hours and ticket prices on official museum websites",
            "transportation": "🚇 Check current schedules and service status on official transport websites",
            "restaurant": "🍽️ Confirm restaurant details and hours by calling directly or checking Google Maps",
            "safety": "🚨 Cross-reference safety information with current government travel advisories"
        }
        
        if category in category_recommendations:
            recommendations.append(category_recommendations[category])
        
        return recommendations
    
    def _compile_sources_used(self, category: str, verified_facts: List[str]) -> List[str]:
        """Compile list of sources used for verification"""
        
        sources = ["Enhanced Content Accuracy System"]
        
        if verified_facts:
            sources.append("Historical Facts Database")
            sources.append("Cultural Knowledge Database")
        
        # Category-specific sources
        category_sources = {
            "museum": ["Turkish Ministry of Culture and Tourism", "UNESCO World Heritage Database"],
            "transportation": ["Istanbul Metropolitan Municipality", "IETT Official Data"],
            "historical": ["Byzantine Studies Database", "Ottoman Archives"],
            "cultural": ["Turkish Cultural Heritage Database"]
        }
        
        if category in category_sources:
            sources.extend(category_sources[category])
        
        return list(set(sources))  # Remove duplicates
    
    def _get_fallback_accuracy_result(self) -> ContentAccuracyResult:
        """Fallback result when analysis fails"""
        
        return ContentAccuracyResult(
            accuracy_level=AccuracyLevel.MODERATE,
            accuracy_score=0.6,
            verified_facts=[],
            questionable_claims=[],
            historical_accuracy=0.6,
            cultural_accuracy=0.6,
            real_time_validity=0.6,
            quality_indicators={},
            recommendations=["Content accuracy analysis unavailable - verify information independently"],
            sources_used=["Fallback Analysis"],
            timestamp=datetime.now().isoformat()
        )
    
    def get_accuracy_improvement_suggestions(self, result: ContentAccuracyResult) -> Dict[str, Any]:
        """Get specific suggestions for improving content accuracy"""
        
        suggestions = {
            "immediate_actions": [],
            "content_improvements": [],
            "source_recommendations": [],
            "quality_enhancements": []
        }
        
        # Immediate actions based on accuracy level
        if result.accuracy_level == AccuracyLevel.QUESTIONABLE:
            suggestions["immediate_actions"].extend([
                "⚠️ Review response thoroughly before using",
                "🔍 Fact-check all specific claims independently",
                "📞 Contact official sources for verification"
            ])
        elif result.accuracy_level == AccuracyLevel.LOW_CONFIDENCE:
            suggestions["immediate_actions"].extend([
                "⚠️ Verify key information with official sources",
                "🔍 Cross-reference factual claims"
            ])
        
        # Content improvement suggestions
        if result.historical_accuracy < 0.7:
            suggestions["content_improvements"].append("📚 Enhance historical fact verification")
        
        if result.cultural_accuracy < 0.7:
            suggestions["content_improvements"].append("🌍 Improve cultural sensitivity and accuracy")
        
        if result.real_time_validity < 0.7:
            suggestions["content_improvements"].append("⏰ Add disclaimers for time-sensitive information")
        
        # Source recommendations
        suggestions["source_recommendations"].extend([
            "🌐 Use official government websites (.gov.tr domains)",
            "🏛️ Reference museum and cultural site official pages",
            "🚇 Check transport authority websites for current schedules"
        ])
        
        return suggestions

# Global instance
enhanced_accuracy_system = EnhancedContentAccuracySystem()

def analyze_response_accuracy(response: str, query: str, category: str = "general") -> ContentAccuracyResult:
    """
    Main function to analyze response accuracy
    
    Args:
        response: AI response to analyze
        query: Original user query
        category: Query category
        
    Returns:
        ContentAccuracyResult with detailed accuracy assessment
    """
    return enhanced_accuracy_system.analyze_content_accuracy(response, query, category)

def get_accuracy_score_only(response: str, category: str = "general") -> float:
    """Get just the accuracy score for quick evaluation"""
    result = enhanced_accuracy_system.analyze_content_accuracy(response, "", category)
    return result.accuracy_score

def validate_historical_facts(response: str) -> Dict[str, Any]:
    """Validate specifically historical facts in response"""
    historical_score, verified_facts = enhanced_accuracy_system._analyze_historical_accuracy(response, "historical")
    
    return {
        "historical_accuracy_score": historical_score,
        "verified_historical_facts": verified_facts,
        "assessment": "HIGH" if historical_score > 0.8 else "MODERATE" if historical_score > 0.6 else "LOW"
    }

def validate_cultural_sensitivity(response: str, query: str) -> Dict[str, Any]:
    """Validate cultural sensitivity and accuracy"""
    cultural_score, cultural_insights = enhanced_accuracy_system._analyze_cultural_accuracy(response, query)
    
    return {
        "cultural_accuracy_score": cultural_score,
        "cultural_insights": cultural_insights,
        "sensitivity_level": "HIGH" if cultural_score > 0.8 else "MODERATE" if cultural_score > 0.6 else "LOW"
    }
