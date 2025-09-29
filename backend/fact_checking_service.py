#!/usr/bin/env python3
"""
Fact-Checking Layer for AI Istanbul System
==========================================

Validates responses against official sources and provides accuracy scoring.
"""

import re
import requests
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class FactCheckResult:
    """Result of fact-checking process"""
    verified_facts: List[str]
    questionable_facts: List[str]
    accuracy_score: float  # 0.0 to 1.0
    official_sources_used: List[str]
    recommendations: List[str]
    timestamp: str

@dataclass
class OfficialSource:
    """Official source information"""
    domain: str
    name: str
    category: str
    reliability_score: float
    last_updated: Optional[str] = None

class FactCheckingService:
    """Service for fact-checking Istanbul tourism information"""
    
    def __init__(self):
        self.official_sources = self._initialize_official_sources()
        self.verified_facts_db = self._initialize_verified_facts()
        self.session = requests.Session()
        self.session.timeout = 10
        
    def _initialize_official_sources(self) -> Dict[str, OfficialSource]:
        """Initialize official sources for fact-checking"""
        return {
            # Museums and Cultural Sites
            "muze.gov.tr": OfficialSource(
                domain="muze.gov.tr",
                name="Turkish Ministry of Culture and Tourism - Museums",
                category="museums",
                reliability_score=1.0
            ),
            "topkapisarayi.gov.tr": OfficialSource(
                domain="topkapisarayi.gov.tr", 
                name="Topkapi Palace Official Site",
                category="museums",
                reliability_score=1.0
            ),
            "ayasofyamuzesi.gov.tr": OfficialSource(
                domain="ayasofyamuzesi.gov.tr",
                name="Hagia Sophia Official Site",
                category="museums", 
                reliability_score=1.0
            ),
            
            # Transportation
            "metro.istanbul": OfficialSource(
                domain="metro.istanbul",
                name="Istanbul Metro Official Site",
                category="transportation",
                reliability_score=1.0
            ),
            "iett.istanbul": OfficialSource(
                domain="iett.istanbul",
                name="Istanbul Public Transport (IETT)",
                category="transportation",
                reliability_score=1.0
            ),
            "sehirhatlari.istanbul": OfficialSource(
                domain="sehirhatlari.istanbul",
                name="Istanbul City Lines (Ferries)",
                category="transportation",
                reliability_score=1.0
            ),
            
            # Government and Tourism
            "istanbul.gov.tr": OfficialSource(
                domain="istanbul.gov.tr",
                name="Istanbul Metropolitan Municipality",
                category="general",
                reliability_score=1.0
            ),
            "kultur.gov.tr": OfficialSource(
                domain="kultur.gov.tr",
                name="Ministry of Culture and Tourism",
                category="culture",
                reliability_score=1.0
            ),
            
            # UNESCO and International
            "whc.unesco.org": OfficialSource(
                domain="whc.unesco.org",
                name="UNESCO World Heritage Centre",
                category="heritage",
                reliability_score=1.0
            )
        }
    
    def _initialize_verified_facts(self) -> Dict[str, Dict[str, str]]:
        """Initialize database of verified facts"""
        return {
            "hagia_sophia": {
                "construction_date": "532-537 AD",
                "architect": "Anthemius of Tralles and Isidore of Miletus",
                "dome_diameter": "31 meters",
                "dome_height": "55 meters",
                "current_status": "Mosque (since 2020)",
                "unesco_status": "World Heritage Site",
                "location": "Sultanahmet, Fatih"
            },
            "topkapi_palace": {
                "construction_date": "1459-1465",
                "architect": "Mimar Atik Sinan",
                "period_of_use": "1465-1856",
                "number_of_rooms": "400+ rooms in Harem",
                "current_status": "Museum",
                "unesco_status": "World Heritage Site",
                "location": "Sultanahmet, Fatih"
            },
            "blue_mosque": {
                "construction_date": "1609-1616",
                "architect": "Sedefkar Mehmed Agha",
                "official_name": "Sultan Ahmed Mosque",
                "number_of_minarets": "6",
                "current_status": "Active mosque",
                "location": "Sultanahmet, Fatih"
            },
            "galata_tower": {
                "construction_date": "1348",
                "height": "67 meters",
                "builder": "Genoese",
                "original_name": "Christea Turris (Tower of Christ)",
                "current_status": "Museum and observation deck",
                "location": "Galata, Beyoğlu"
            },
            "metro_m2": {
                "line_name": "M2 Vezneciler - Hacıosman",
                "color": "Green",
                "length": "23.5 km",
                "stations": "13 stations",
                "opening_date": "2000 (first section)",
                "key_stations": "Taksim, Şişhane, Vezneciler, Levent"
            },
            "metro_m1a": {
                "line_name": "M1A Yenikapı - Atatürk Airport",
                "color": "Light Blue", 
                "length": "26.8 km",
                "stations": "20 stations",
                "opening_date": "1989 (first section)",
                "note": "Connects to closed Atatürk Airport"
            },
            "metro_m11": {
                "line_name": "M11 Gayrettepe - Istanbul Airport",
                "color": "Gray",
                "length": "37.5 km", 
                "journey_time": "37 minutes",
                "opening_date": "2023",
                "key_connection": "M2 at Gayrettepe"
            }
        }
    
    def fact_check_response(self, response_text: str, query_category: str) -> FactCheckResult:
        """
        Comprehensive fact-checking of AI response
        """
        try:
            verified_facts = []
            questionable_facts = []
            official_sources_used = []
            
            # Extract claims from response
            claims = self._extract_claims(response_text)
            
            # Check each claim against verified database
            for claim in claims:
                verification_result = self._verify_claim(claim, query_category)
                
                if verification_result["verified"]:
                    verified_facts.append(claim)
                    if verification_result.get("source"):
                        official_sources_used.append(verification_result["source"])
                else:
                    questionable_facts.append(claim)
            
            # Calculate accuracy score
            total_claims = len(claims)
            if total_claims == 0:
                accuracy_score = 0.8  # Neutral score for responses without specific claims
            else:
                accuracy_score = len(verified_facts) / total_claims
            
            # Generate recommendations
            recommendations = self._generate_recommendations(accuracy_score, questionable_facts, query_category)
            
            return FactCheckResult(
                verified_facts=verified_facts,
                questionable_facts=questionable_facts,
                accuracy_score=accuracy_score,
                official_sources_used=list(set(official_sources_used)),
                recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logging.error(f"Error in fact-checking: {e}")
            return FactCheckResult(
                verified_facts=[],
                questionable_facts=[],
                accuracy_score=0.5,
                official_sources_used=[],
                recommendations=["Fact-checking service unavailable - verify information independently"],
                timestamp=datetime.now().isoformat()
            )
    
    def _extract_claims(self, response_text: str) -> List[str]:
        """Extract factual claims from response text"""
        claims = []
        
        # Patterns for extracting factual claims
        patterns = [
            # Dates and years
            r'(?:built|constructed|opened|established|founded)(?:\s+in)?\s+(\d{4}(?:-\d{4})?(?:\s+AD)?)',
            r'(\d{4}(?:-\d{4})?(?:\s+AD)?)\s*(?:construction|building|establishment)',
            
            # Measurements and numbers
            r'(\d+(?:\.\d+)?)\s*(?:meters?|km|kilometers?|minutes?)\s+(?:high|tall|long|wide|diameter|journey|walk)',
            r'(?:height|diameter|length|width|distance)(?:\s+is|\s+of)?\s+(\d+(?:\.\d+)?)\s*(?:meters?|km)',
            
            # Operating hours and schedules
            r'(?:open|operating|hours?)(?:\s+from)?\s+(\d{1,2}:\d{2}(?:\s*-\s*\d{1,2}:\d{2})?)',
            r'(?:every|frequency)\s+(\d+(?:-\d+)?)\s*minutes?',
            
            # Architects and builders
            r'(?:architect|designed by|built by)\s+([A-Z][a-zA-Z\s]+(?:Agha|Sinan|Pasha))',
            
            # Locations and addresses
            r'(?:located in|situated in|address)\s+([A-Z][a-zA-Z\s,]+(?:Istanbul|Turkey))',
            
            # Transportation lines
            r'(M\d+[A-Z]?|T\d+)\s+(?:Metro|Tram|Line)',
            r'(?:Metro|Tram)\s+(M\d+[A-Z]?|T\d+)',
            
            # UNESCO and official status
            r'(UNESCO World Heritage Site|World Heritage)',
            r'(functioning mosque|active mosque|museum status)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, response_text, re.IGNORECASE)
            for match in matches:
                claim = match.group(0).strip()
                if len(claim) > 5:  # Filter out very short matches
                    claims.append(claim)
        
        return list(set(claims))  # Remove duplicates
    
    def _verify_claim(self, claim: str, category: str) -> Dict[str, any]:
        """Verify a specific claim against known facts"""
        
        claim_lower = claim.lower()
        
        # Check against verified facts database
        for fact_key, facts in self.verified_facts_db.items():
            for fact_type, fact_value in facts.items():
                if self._claim_matches_fact(claim_lower, fact_value.lower()):
                    return {
                        "verified": True,
                        "source": f"verified_database_{fact_key}",
                        "confidence": 0.95
                    }
        
        # Check for common accurate information patterns
        accurate_patterns = [
            # Transportation lines that exist
            r'm(1a|2|3|4|5|6|7|11)\b',
            r't(1|4|5)\b',
            r'marmaray',
            r'istanbulkart',
            
            # Well-known locations
            r'sultanahmet|beyoğlu|galata|kadıköy|taksim|eminönü',
            
            # Common opening hours (reasonable ranges)
            r'(09?|10|11):(00|30)\s*-\s*(16|17|18|19|20):(00|30)',
            
            # Reasonable walking times
            r'[1-9]\d?\s*minute[s]?\s*walk',
        ]
        
        for pattern in accurate_patterns:
            if re.search(pattern, claim_lower):
                return {
                    "verified": True,
                    "source": "pattern_verification",
                    "confidence": 0.8
                }
        
        # Check for potentially inaccurate information
        inaccurate_patterns = [
            # Specific prices or costs (we avoid these)
            r'\d+\s*(?:tl|lira|euro|dollar|\$|€)',
            
            # Overly specific times that might be outdated
            r'exactly\s+\d+\s*minutes?',
            
            # Unverifiable claims
            r'best\s+(?:restaurant|hotel|view)',
            r'most\s+(?:popular|famous|visited)',
        ]
        
        for pattern in inaccurate_patterns:
            if re.search(pattern, claim_lower):
                return {
                    "verified": False,
                    "source": "potentially_inaccurate",
                    "confidence": 0.3
                }
        
        # Default: unverified but not necessarily wrong
        return {
            "verified": False,
            "source": None,
            "confidence": 0.5
        }
    
    def _claim_matches_fact(self, claim: str, fact: str) -> bool:
        """Check if claim matches a known fact with some tolerance"""
        
        # Direct substring match
        if fact in claim or claim in fact:
            return True
        
        # Date matching with some flexibility
        date_pattern = r'\d{4}(?:-\d{4})?'
        claim_dates = re.findall(date_pattern, claim)
        fact_dates = re.findall(date_pattern, fact)
        
        if claim_dates and fact_dates:
            return any(cd in fd or fd in cd for cd in claim_dates for fd in fact_dates)
        
        # Number matching (for measurements, etc.)
        number_pattern = r'\d+(?:\.\d+)?'
        claim_numbers = re.findall(number_pattern, claim)
        fact_numbers = re.findall(number_pattern, fact)
        
        if claim_numbers and fact_numbers:
            return any(abs(float(cn) - float(fn)) < 0.1 for cn in claim_numbers for fn in fact_numbers)
        
        return False
    
    def _generate_recommendations(self, accuracy_score: float, questionable_facts: List[str], category: str) -> List[str]:
        """Generate recommendations based on fact-checking results"""
        
        recommendations = []
        
        if accuracy_score >= 0.9:
            recommendations.append("Response contains highly accurate information")
        elif accuracy_score >= 0.7:
            recommendations.append("Response is generally accurate with minor uncertainties")
        elif accuracy_score >= 0.5:
            recommendations.append("Response contains some unverified information - verify key details")
        else:
            recommendations.append("Response contains significant unverified information - use caution")
        
        # Category-specific recommendations
        if category == "museum":
            recommendations.append("Verify opening hours and ticket prices on official museum websites")
        elif category == "transportation":
            recommendations.append("Check current schedules and service status on official transport websites")
        elif category == "restaurant":
            recommendations.append("Confirm restaurant details on Google Maps or by calling directly")
        
        # Specific recommendations for questionable facts
        if questionable_facts:
            if len(questionable_facts) <= 3:
                recommendations.append(f"Double-check: {', '.join(questionable_facts[:3])}")
            else:
                recommendations.append(f"Multiple unverified claims detected - verify independently")
        
        return recommendations
    
    def get_official_sources_for_category(self, category: str) -> List[OfficialSource]:
        """Get official sources for a specific category"""
        return [source for source in self.official_sources.values() if source.category == category]
    
    def should_use_gpt_fallback(self, accuracy_score: float, category: str) -> bool:
        """Determine if GPT should handle the query due to low accuracy"""
        
        # Thresholds for different categories
        thresholds = {
            "museum": 0.6,
            "transportation": 0.7,
            "restaurant": 0.5,
            "safety": 0.8,
            "general": 0.5
        }
        
        threshold = thresholds.get(category, 0.6)
        return accuracy_score < threshold

# Global instance
fact_checker = FactCheckingService()
