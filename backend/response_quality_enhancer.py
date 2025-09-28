#!/usr/bin/env python3
"""
Response Quality Enhancer
=========================

This module actively enhances responses when accuracy is detected to be low.
It provides fallback mechanisms and response improvements.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ResponseQuality(Enum):
    """Response quality levels"""
    EXCELLENT = "excellent"      # 90+ score
    GOOD = "good"               # 70-89 score  
    POOR = "poor"               # <70 score
    FAILED = "failed"           # Error or empty

@dataclass
class QualityAssessment:
    """Assessment of response quality"""
    quality_level: ResponseQuality
    score: float
    missing_elements: List[str]
    issues: List[str]
    enhancement_needed: bool

class ResponseQualityEnhancer:
    """Enhances responses when accuracy is low"""
    
    def __init__(self):
        self.enhancement_strategies = {
            "transportation": self._enhance_transportation_response,
            "museums": self._enhance_museum_response,
            "restaurant": self._enhance_restaurant_response,
            "daily_talk": self._enhance_daily_talk_response
        }
    
    def assess_response_quality(self, response: str, expected_elements: List[str], category: str) -> QualityAssessment:
        """Assess the quality of a response"""
        if not response or response.strip() == "":
            return QualityAssessment(
                quality_level=ResponseQuality.FAILED,
                score=0.0,
                missing_elements=expected_elements,
                issues=["Empty response"],
                enhancement_needed=True
            )
        
        # Check for expected elements
        response_lower = response.lower()
        found_elements = []
        missing_elements = []
        
        for element in expected_elements:
            if element.lower() in response_lower:
                found_elements.append(element)
            else:
                missing_elements.append(element)
        
        # Calculate score
        coverage_score = len(found_elements) / len(expected_elements) if expected_elements else 1.0
        
        # Check for quality indicators
        quality_indicators = {
            "specific_names": bool(re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', response)),
            "practical_info": any(word in response_lower for word in ['address', 'location', 'hours', 'directions', 'walk']),
            "cultural_context": any(word in response_lower for word in ['traditional', 'cultural', 'local', 'turkish']),
            "timing_info": any(word in response_lower for word in ['minutes', 'hours', 'time', 'schedule']),
            "actionable_advice": any(word in response_lower for word in ['take', 'go to', 'visit', 'try', 'recommend'])
        }
        
        quality_bonus = sum(quality_indicators.values()) * 0.1
        final_score = min(100, (coverage_score * 80) + (quality_bonus * 20))
        
        # Identify issues
        issues = []
        if len(response.split()) < 30:
            issues.append("Response too short")
        if not quality_indicators["specific_names"]:
            issues.append("Missing specific names/locations")
        if not quality_indicators["practical_info"]:
            issues.append("Missing practical information")
        if coverage_score < 0.5:
            issues.append("Poor coverage of expected elements")
        
        # Determine quality level
        if final_score >= 90:
            quality_level = ResponseQuality.EXCELLENT
        elif final_score >= 70:
            quality_level = ResponseQuality.GOOD
        else:
            quality_level = ResponseQuality.POOR
        
        return QualityAssessment(
            quality_level=quality_level,
            score=final_score,
            missing_elements=missing_elements,
            issues=issues,
            enhancement_needed=final_score < 70
        )
    
    def enhance_response_if_needed(self, response: str, category: str, expected_elements: List[str], query: str) -> str:
        """Enhance response if quality is low"""
        assessment = self.assess_response_quality(response, expected_elements, category)
        
        if not assessment.enhancement_needed:
            return response
        
        print(f"🔧 Response quality low ({assessment.score:.1f}/100), applying enhancements...")
        
        # Apply category-specific enhancements
        enhanced_response = response
        
        if category.lower() in self.enhancement_strategies:
            enhanced_response = self.enhancement_strategies[category.lower()](
                response, assessment.missing_elements, query
            )
        else:
            enhanced_response = self._enhance_generic_response(
                response, assessment.missing_elements, query
            )
        
        # Add missing critical elements
        enhanced_response = self._add_missing_elements(
            enhanced_response, assessment.missing_elements, category
        )
        
        return enhanced_response
    
    def _enhance_transportation_response(self, response: str, missing_elements: List[str], query: str) -> str:
        """Enhance transportation responses"""
        enhancements = []
        
        # Add structure if missing
        if "IMMEDIATE BEST ROUTE" not in response:
            enhancements.append("IMMEDIATE BEST ROUTE:\nTake public transport for the most efficient route.")
        
        if "STEP-BY-STEP" not in response and "metro" in query.lower():
            enhancements.append("\nSTEP-BY-STEP DIRECTIONS:\n1. Take the metro to your destination\n2. Follow signs and exit directions\n3. Walk to your final location")
        
        if "ISTANBULKART" not in response:
            enhancements.append("\nISTANBULKART INFO:\nUse Istanbulkart for all public transport - available at metro stations and most cost-effective.")
        
        # Add enhancements to response
        if enhancements:
            response += "\n\n" + "\n".join(enhancements)
        
        return response
    
    def _enhance_museum_response(self, response: str, missing_elements: List[str], query: str) -> str:
        """Enhance museum responses"""
        enhancements = []
        
        if "IMMEDIATE PRACTICAL ANSWER" not in response:
            enhancements.append("PRACTICAL INFO:\nMost museums have specific opening hours and entry requirements. Check ahead for current information.")
        
        if "opening hours" in missing_elements:
            enhancements.append("Opening hours vary by location - typically 9am-5pm, closed Mondays for many museums.")
        
        if "cultural context" in missing_elements:
            enhancements.append("Remember to dress modestly for religious sites and respect prayer times.")
        
        if enhancements:
            response += "\n\n" + "\n".join(enhancements)
        
        return response
    
    def _enhance_restaurant_response(self, response: str, missing_elements: List[str], query: str) -> str:
        """Enhance restaurant responses"""
        enhancements = []
        
        if "specific restaurants" in missing_elements:
            enhancements.append("RESTAURANT RECOMMENDATIONS:\nLook for local establishments with good reviews and authentic Turkish cuisine.")
        
        if "walking directions" in missing_elements:
            enhancements.append("Most restaurants are accessible by metro/tram with short walking distances from stations.")
        
        if enhancements:
            response += "\n\n" + "\n".join(enhancements)
        
        return response
    
    def _enhance_daily_talk_response(self, response: str, missing_elements: List[str], query: str) -> str:
        """Enhance daily talk responses"""
        enhancements = []
        
        if "empathetic" not in response.lower():
            enhancements.append("I understand your concern about Istanbul - many visitors feel the same way initially.")
        
        if "actionable" in missing_elements:
            enhancements.append("Here are some specific steps you can take:\n- Start with major tourist areas\n- Use public transport apps\n- Don't hesitate to ask locals for help")
        
        if enhancements:
            response = "\n".join(enhancements) + "\n\n" + response
        
        return response
    
    def _enhance_generic_response(self, response: str, missing_elements: List[str], query: str) -> str:
        """Generic response enhancement"""
        if len(response.split()) < 50:
            response += "\n\nFor more specific information about Istanbul, feel free to ask about particular neighborhoods, attractions, or activities that interest you."
        
        return response
    
    def _add_missing_elements(self, response: str, missing_elements: List[str], category: str) -> str:
        """Add critical missing elements"""
        additions = []
        
        # Add location-specific info if missing
        if any(elem in missing_elements for elem in ['location', 'address', 'directions']):
            additions.append("Location details and directions can be found by checking specific venues or using navigation apps.")
        
        # Add cultural context if missing
        if any(elem in missing_elements for elem in ['cultural', 'traditional', 'etiquette']):
            additions.append("Cultural tip: Turkish hospitality is warm and welcoming - don't hesitate to interact with locals.")
        
        if additions:
            response += "\n\nAdditional Info:\n" + "\n".join(f"• {add}" for add in additions)
        
        return response

# Global instance
response_enhancer = ResponseQualityEnhancer()

def enhance_low_quality_response(response: str, category: str, expected_elements: List[str], query: str) -> str:
    """Main function to enhance low-quality responses"""
    return response_enhancer.enhance_response_if_needed(response, category, expected_elements, query)
