#!/usr/bin/env python3
"""
Enhanced Feature Detection System
=================================

This module provides sophisticated feature detection to evaluate chatbot responses
and ensure they meet quality standards for completeness and relevance.

Target improvements:
- Feature Coverage: 26.6% -> 80%+
- More accurate evaluation of response quality
- Category-specific feature detection
- Better alignment with user expectations
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass

class FeatureCategory(Enum):
    """Categories of features to detect in responses"""
    BASIC_INFO = "basic_info"
    LOCATION_SPECIFIC = "location_specific"
    PRACTICAL_DETAILS = "practical_details"
    CULTURAL_CONTEXT = "cultural_context"
    TRANSPORTATION = "transportation"
    TIMING_INFO = "timing_info"
    SAFETY_GUIDANCE = "safety_guidance"
    INSIDER_KNOWLEDGE = "insider_knowledge"

@dataclass
class DetectedFeature:
    """Represents a detected feature in a response"""
    feature_id: str
    category: FeatureCategory
    confidence: float
    evidence: str
    weight: float = 1.0

class EnhancedFeatureDetector:
    """Advanced feature detection for chatbot response evaluation"""
    
    def __init__(self):
        self.feature_patterns = self._build_feature_patterns()
        self.location_keywords = self._build_location_keywords()
        self.cultural_keywords = self._build_cultural_keywords()
        self.practical_keywords = self._build_practical_keywords()
        
    def _build_feature_patterns(self) -> Dict[str, Dict[str, any]]:
        """Build comprehensive feature detection patterns"""
        return {
            # Daily Talk Features
            "welcoming_tone": {
                "patterns": [r'\b(welcome|merhaba|hello|hi there)\b', r'\b(great question|happy to help|glad you asked)\b'],
                "category": FeatureCategory.BASIC_INFO,
                "weight": 1.0
            },
            "practical_advice": {
                "patterns": [r'\btip\b', r'\badvi[sc]e\b', r'\brecommend\b', r'\bsuggestion\b', r'\bhelpful\b'],
                "category": FeatureCategory.PRACTICAL_DETAILS,
                "weight": 2.0
            },
            "cultural_tips": {
                "patterns": [r'\betiquette\b', r'\bcustoms?\b', r'\btradition\b', r'\bcultural\b', r'\brespect\b'],
                "category": FeatureCategory.CULTURAL_CONTEXT,
                "weight": 2.0
            },
            "specific_locations": {
                "patterns": [r'\b(sultanahmet|beyoğlu|kadıköy|taksim|galata|eminönü|balat)\b'],
                "category": FeatureCategory.LOCATION_SPECIFIC,
                "weight": 2.0
            },
            "transportation_info": {
                "patterns": [r'\b(metro|tram|bus|ferry|taxi|istanbulkart)\b', r'\bget to\b', r'\btransportation\b'],
                "category": FeatureCategory.TRANSPORTATION,
                "weight": 1.5
            },
            "time_context": {
                "patterns": [r'\bbest time\b', r'\bearly morning\b', r'\bevening\b', r'\bweekend\b', r'\bhours?\b'],
                "category": FeatureCategory.TIMING_INFO,
                "weight": 1.5
            },
            "safety_guidance": {
                "patterns": [r'\bsafe\b', r'\bsecurity\b', r'\bavoid\b', r'\bcareful\b', r'\baware\b'],
                "category": FeatureCategory.SAFETY_GUIDANCE,
                "weight": 2.0
            },
            "insider_knowledge": {
                "patterns": [r'\blocals?\b', r'\bhidden\b', r'\bsecret\b', r'\boff the beaten\b', r'\bauthenti[ck]\b'],
                "category": FeatureCategory.INSIDER_KNOWLEDGE,
                "weight": 2.5
            },
            
            # Restaurant Features
            "specific_restaurants": {
                "patterns": [r'\b[A-Z][a-z]+\s+(restaurant|lokanta|meyhane)\b', r'\bnamed?\s+\w+\b.*restaurant'],
                "category": FeatureCategory.LOCATION_SPECIFIC,
                "weight": 3.0
            },
            "signature_dishes": {
                "patterns": [r'\b(döner|kebab|baklava|börek|meze|turkish breakfast|kahvaltı)\b', r'\bmust try\b', r'\bspecialty\b'],
                "category": FeatureCategory.PRACTICAL_DETAILS,
                "weight": 2.0
            },
            "atmosphere_description": {
                "patterns": [r'\batmosphere\b', r'\bambiance\b', r'\bromantic\b', r'\bcasual\b', r'\bupscale\b', r'\bfamily-friendly\b'],
                "category": FeatureCategory.PRACTICAL_DETAILS,
                "weight": 1.5
            },
            "location_details": {
                "patterns": [r'\baddress\b', r'\bstreet\b', r'\bnear\b', r'\baround\b', r'\bwalk from\b'],
                "category": FeatureCategory.LOCATION_SPECIFIC,
                "weight": 2.0
            },
            "price_context": {
                "patterns": [r'\baffordable\b', r'\bbudget\b', r'\bmoderate\b', r'\bupscale\b', r'\bprice\b'],
                "category": FeatureCategory.PRACTICAL_DETAILS,
                "weight": 1.5
            },
            
            # District Features
            "district_character": {
                "patterns": [r'\bcharacter\b', r'\bknown for\b', r'\bfamous for\b', r'\bunique\b', r'\batmosphere\b'],
                "category": FeatureCategory.CULTURAL_CONTEXT,
                "weight": 2.0
            },
            "key_attractions": {
                "patterns": [r'\bmain attractions?\b', r'\bmust see\b', r'\bhighlights?\b', r'\blandmarks?\b'],
                "category": FeatureCategory.PRACTICAL_DETAILS,
                "weight": 2.0
            },
            "walking_routes": {
                "patterns": [r'\bwalk\b', r'\bminutes?\b', r'\broute\b', r'\bdistance\b', r'\bsteps?\b'],
                "category": FeatureCategory.TRANSPORTATION,
                "weight": 2.0
            },
            
            # Museum Features
            "historical_significance": {
                "patterns": [r'\bhistory\b', r'\bhistorical\b', r'\bbyzantine\b', r'\bottoman\b', r'\bcentur\w+\b'],
                "category": FeatureCategory.CULTURAL_CONTEXT,
                "weight": 2.5
            },
            "key_highlights": {
                "patterns": [r'\bhighlights?\b', r'\bmust see\b', r'\bexhibits?\b', r'\bcollection\b', r'\bmasterpiece\b'],
                "category": FeatureCategory.PRACTICAL_DETAILS,
                "weight": 2.0
            },
            "visiting_strategies": {
                "patterns": [r'\bbest time to visit\b', r'\bavoid crowds\b', r'\btiming\b', r'\bstrategy\b'],
                "category": FeatureCategory.TIMING_INFO,
                "weight": 2.0
            },
            
            # Transportation Features  
            "specific_routes": {
                "patterns": [r'\bM\d+\b', r'\bT\d+\b', r'\bline\b', r'\broute\b', r'\bstation\b'],
                "category": FeatureCategory.TRANSPORTATION,
                "weight": 2.5
            },
            "step_by_step_directions": {
                "patterns": [r'\b(first|then|next|finally)\b', r'\bstep\s+\d+\b', r'\bdirections?\b'],
                "category": FeatureCategory.PRACTICAL_DETAILS,
                "weight": 2.5
            },
            "istanbulkart_info": {
                "patterns": [r'\bistanbulkart\b', r'\bcard\b.*\btransport\b', r'\btop up\b', r'\brecharge\b'],
                "category": FeatureCategory.PRACTICAL_DETAILS,
                "weight": 2.0
            },
            "alternative_routes": {
                "patterns": [r'\balternative\b', r'\balso\b', r'\bor you can\b', r'\banother option\b'],
                "category": FeatureCategory.PRACTICAL_DETAILS,
                "weight": 1.5
            }
        }
    
    def _build_location_keywords(self) -> Set[str]:
        """Build set of Istanbul location keywords"""
        return {
            'sultanahmet', 'beyoğlu', 'beyoglu', 'kadıköy', 'kadikoy', 'taksim', 'galata',
            'eminönü', 'eminonu', 'balat', 'fener', 'ortaköy', 'ortakoy', 'beşiktaş', 'besiktas',
            'üsküdar', 'uskudar', 'fatih', 'şişli', 'sisli', 'levent', 'bosphorus', 'golden horn',
            'hagia sophia', 'blue mosque', 'topkapi', 'galata tower', 'grand bazaar', 'spice bazaar',
            'istiklal', 'tünel', 'tunel', 'karaköy', 'karakoy', 'moda', 'cihangir'
        }
    
    def _build_cultural_keywords(self) -> Set[str]:
        """Build set of cultural context keywords"""
        return {
            'turkish', 'ottoman', 'byzantine', 'islamic', 'tradition', 'culture', 'heritage',
            'etiquette', 'customs', 'respect', 'mosque', 'prayer', 'ramadan', 'halal',
            'merhaba', 'teşekkür', 'lutfen', 'please', 'thank you', 'local', 'authentic'
        }
    
    def _build_practical_keywords(self) -> Set[str]:
        """Build set of practical information keywords"""
        return {
            'hours', 'open', 'closed', 'ticket', 'entrance', 'fee', 'reservation', 'book',
            'crowded', 'busy', 'quiet', 'best time', 'avoid', 'tip', 'advice', 'recommend',
            'metro', 'tram', 'bus', 'ferry', 'taxi', 'walk', 'distance', 'minutes',
            'istanbulkart', 'bitaksi', 'app', 'navigation', 'directions'
        }
    
    def detect_features(self, response: str, query_category: str = "generic") -> Dict[str, DetectedFeature]:
        """
        Detect features in a chatbot response
        
        Returns dictionary of feature_id -> DetectedFeature
        """
        detected_features = {}
        response_lower = response.lower()
        
        for feature_id, feature_config in self.feature_patterns.items():
            confidence = 0.0
            evidence_parts = []
            
            # Pattern-based detection
            for pattern in feature_config["patterns"]:
                matches = re.findall(pattern, response_lower)
                if matches:
                    confidence += len(matches) * 0.3
                    evidence_parts.extend(matches if isinstance(matches[0], str) else [str(m) for m in matches])
            
            # Category-specific enhancements
            if query_category == "restaurant" and feature_id in ["specific_restaurants", "signature_dishes"]:
                confidence *= 1.5
            elif query_category == "district" and feature_id in ["district_character", "key_attractions"]:
                confidence *= 1.5
            elif query_category == "museum" and feature_id in ["historical_significance", "key_highlights"]:
                confidence *= 1.5
            elif query_category == "transportation" and feature_id in ["specific_routes", "istanbulkart_info"]:
                confidence *= 1.5
            
            # Additional scoring based on content depth
            if feature_id == "specific_locations":
                location_count = sum(1 for loc in self.location_keywords if loc in response_lower)
                confidence += location_count * 0.4
                if location_count > 0:
                    evidence_parts.extend([loc for loc in self.location_keywords if loc in response_lower])
            
            if feature_id == "cultural_tips":
                cultural_count = sum(1 for term in self.cultural_keywords if term in response_lower)
                confidence += cultural_count * 0.3
            
            if feature_id == "practical_advice":
                practical_count = sum(1 for term in self.practical_keywords if term in response_lower)
                confidence += practical_count * 0.2
            
            # Response length consideration
            if len(response) > 200:
                confidence *= 1.1
            if len(response) > 400:
                confidence *= 1.2
                
            # Cap confidence at 1.0
            confidence = min(confidence, 1.0)
            
            if confidence > 0.2:  # Threshold for feature detection
                detected_features[feature_id] = DetectedFeature(
                    feature_id=feature_id,
                    category=feature_config["category"],
                    confidence=confidence,
                    evidence=", ".join(evidence_parts[:3]) if evidence_parts else "pattern match",
                    weight=feature_config["weight"]
                )
        
        return detected_features
    
    def calculate_feature_coverage(self, detected_features: Dict[str, DetectedFeature], 
                                 expected_features: List[str]) -> Tuple[float, Dict[str, bool]]:
        """
        Calculate feature coverage percentage and detailed coverage map
        
        Returns:
            coverage_percentage: Percentage of expected features detected (0-100)
            coverage_map: Dictionary showing which expected features were found
        """
        if not expected_features:
            return 100.0, {}
        
        coverage_map = {}
        detected_feature_ids = set(detected_features.keys())
        
        for feature in expected_features:
            coverage_map[feature] = feature in detected_feature_ids
        
        covered_count = sum(coverage_map.values())
        coverage_percentage = (covered_count / len(expected_features)) * 100
        
        return coverage_percentage, coverage_map
    
    def calculate_completeness_score(self, detected_features: Dict[str, DetectedFeature]) -> float:
        """
        Calculate overall completeness score based on detected features and their weights
        
        Returns score from 0.0 to 5.0
        """
        if not detected_features:
            return 0.0
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for feature in detected_features.values():
            weighted_score += feature.confidence * feature.weight
            total_weight += feature.weight
        
        if total_weight == 0:
            return 0.0
            
        # Normalize to 0-5 scale
        normalized_score = (weighted_score / total_weight) * 5.0
        return min(normalized_score, 5.0)
    
    def get_feature_analysis(self, response: str, query_category: str = "generic", 
                           expected_features: List[str] = None) -> Dict[str, any]:
        """
        Comprehensive feature analysis of a response
        
        Returns detailed analysis including:
        - detected_features: Dict of detected features
        - coverage_percentage: Feature coverage percentage
        - completeness_score: Overall completeness score (0-5)
        - missing_features: List of expected but missing features
        - feature_categories: Breakdown by feature category
        """
        if expected_features is None:
            expected_features = []
        
        detected_features = self.detect_features(response, query_category)
        coverage_percentage, coverage_map = self.calculate_feature_coverage(detected_features, expected_features)
        completeness_score = self.calculate_completeness_score(detected_features)
        
        missing_features = [f for f, found in coverage_map.items() if not found]
        
        # Group by category
        feature_categories = {}
        for feature in detected_features.values():
            category = feature.category.value
            if category not in feature_categories:
                feature_categories[category] = []
            feature_categories[category].append(feature.feature_id)
        
        return {
            "detected_features": detected_features,
            "coverage_percentage": coverage_percentage,
            "completeness_score": completeness_score,
            "missing_features": missing_features,
            "feature_categories": feature_categories,
            "coverage_map": coverage_map,
            "total_features_detected": len(detected_features),
            "expected_features_count": len(expected_features)
        }

# Global instance
enhanced_feature_detector = EnhancedFeatureDetector()

def analyze_response_features(response: str, query_category: str = "generic", 
                            expected_features: List[str] = None) -> Dict[str, any]:
    """
    Main function to analyze response features
    
    Usage:
        analysis = analyze_response_features(response, "restaurant", ["specific_restaurants", "signature_dishes"])
        print(f"Completeness: {analysis['completeness_score']:.1f}/5.0")
        print(f"Coverage: {analysis['coverage_percentage']:.1f}%")
    """
    return enhanced_feature_detector.get_feature_analysis(response, query_category, expected_features)
