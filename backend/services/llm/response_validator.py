"""
Response Validation Layer

Post-generation validation for LLM responses.
Detects:
- Hallucinations (facts not in context)
- Contradictions (conflicts with verified data)
- Template-like responses (generic/not grounded)
- Over-confident claims

Can trigger:
- Self-repair (ask LLM to fix)
- Hedging (add uncertainty markers)
- Partial abort (stop streaming early)

Author: AI Istanbul Team
Date: December 2024
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationIssue(Enum):
    """Types of validation issues"""
    HALLUCINATION = "hallucination"          # Claims not in context
    CONTRADICTION = "contradiction"          # Conflicts with verified data
    TEMPLATE_RESPONSE = "template_response"  # Generic/boilerplate
    OVER_CONFIDENT = "over_confident"        # Too certain without grounding
    WRONG_LANGUAGE = "wrong_language"        # Response in wrong language
    PROMPT_LEAK = "prompt_leak"              # System prompt leaked
    INJECTION_DETECTED = "injection_detected" # Prompt injection attempted


class ValidationSeverity(Enum):
    """Severity of validation issues"""
    LOW = "low"           # Warning only
    MEDIUM = "medium"     # Add hedging
    HIGH = "high"         # Attempt self-repair
    CRITICAL = "critical" # Abort and fallback


@dataclass
class ValidationResult:
    """Result of response validation"""
    is_valid: bool
    issues: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 1.0
    suggested_action: str = "none"  # none, hedge, repair, abort
    hedging_additions: List[str] = field(default_factory=list)
    repaired_response: Optional[str] = None
    
    def get_worst_severity(self) -> ValidationSeverity:
        """Get the worst severity among all issues"""
        if not self.issues:
            return ValidationSeverity.LOW
        
        severity_order = [ValidationSeverity.LOW, ValidationSeverity.MEDIUM, 
                         ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]
        worst = ValidationSeverity.LOW
        
        for issue in self.issues:
            severity = issue.get('severity', ValidationSeverity.LOW)
            if severity_order.index(severity) > severity_order.index(worst):
                worst = severity
        
        return worst


class ResponseValidator:
    """
    Validates LLM responses against grounded context.
    
    Key responsibilities:
    1. Detect facts not present in context (hallucinations)
    2. Find contradictions with verified data
    3. Identify template/generic responses
    4. Suggest repairs or hedging
    """
    
    # Template phrases that indicate generic responses
    TEMPLATE_INDICATORS = [
        "I'd be happy to help",
        "as an AI",
        "I don't have access to",
        "based on my training",
        "I apologize, but I'm unable to",
        "According to my knowledge",
        "As of my last update",
        "I recommend consulting",
        "For the most accurate information",
        "please note that",
        "it's important to note",
        "keep in mind that",
        "I should mention that",
        "I want to clarify that",
        "as a language model",
        "I cannot provide real-time",
    ]
    
    # Prompt leak indicators
    PROMPT_LEAK_PATTERNS = [
        r"critical\s+rules?\s*[:\-]?\s*follow",
        r"security\s*[:\-]?\s*ignore",
        r"response\s+format\s*[:\-]",
        r"accuracy\s+rules?\s*[:\-]",
        r"you are\s+(kam|an?\s+expert|a\s+guide)",
        r"grounding\s+instructions?",
        r"context\s+assembly",
        r"immutable\s+facts?",
        r"verified\s+route\s+data",
        r"fact\s+layer",
        r"reasoning\s+layer",
    ]
    
    # Over-confident phrases without grounding
    OVER_CONFIDENT_PHRASES = [
        "definitely",
        "certainly",
        "absolutely",
        "without a doubt",
        "100%",
        "guaranteed",
        "always",
        "never fails",
        "the best",
        "perfect",
        "exactly right",
    ]
    
    # Hedging phrases we can add
    HEDGING_PHRASES = {
        'en': [
            "Based on available information, ",
            "According to our data, ",
            "From what we have, ",
            "The information suggests that ",
        ],
        'tr': [
            "Mevcut bilgilere göre, ",
            "Verilerimize göre, ",
            "Elimizdeki bilgiye göre, ",
        ],
    }
    
    def __init__(self):
        # Compile regex patterns
        self._prompt_leak_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PROMPT_LEAK_PATTERNS
        ]
    
    def validate(
        self,
        response: str,
        context_sources: List[Dict[str, Any]],
        expected_language: str = "en",
        route_data: Optional[Dict[str, Any]] = None,
        grounding_contract: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate an LLM response against grounded context.
        
        Args:
            response: The generated response text
            context_sources: List of context items with source and content
            expected_language: Expected response language
            route_data: Immutable route data (if transportation query)
            grounding_contract: Grounding instructions from context assembly
            
        Returns:
            ValidationResult with issues and suggested actions
        """
        issues = []
        confidence = 1.0
        
        # 1. Check for prompt leaks
        leak_issues = self._check_prompt_leaks(response)
        issues.extend(leak_issues)
        if leak_issues:
            confidence -= 0.3
        
        # 2. Check for template responses
        template_issues = self._check_template_response(response)
        issues.extend(template_issues)
        if template_issues:
            confidence -= 0.2
        
        # 3. Check language consistency
        lang_issues = self._check_language(response, expected_language)
        issues.extend(lang_issues)
        if lang_issues:
            confidence -= 0.15
        
        # 4. Check route data consistency (if applicable)
        if route_data:
            route_issues = self._check_route_consistency(response, route_data)
            issues.extend(route_issues)
            if route_issues:
                confidence -= 0.25
        
        # 5. Check for hallucinated facts
        halluc_issues = self._check_hallucinations(response, context_sources)
        issues.extend(halluc_issues)
        if halluc_issues:
            confidence -= 0.2 * len(halluc_issues)
        
        # 6. Check over-confidence
        overconf_issues = self._check_over_confidence(response, context_sources)
        issues.extend(overconf_issues)
        if overconf_issues:
            confidence -= 0.1
        
        # Determine action
        result = ValidationResult(
            is_valid=len([i for i in issues if i['severity'] in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]]) == 0,
            issues=issues,
            confidence_score=max(0.0, confidence),
            suggested_action=self._determine_action(issues)
        )
        
        # Add hedging suggestions if needed
        if result.suggested_action == "hedge":
            result.hedging_additions = self._generate_hedging(expected_language)
        
        return result
    
    def _check_prompt_leaks(self, response: str) -> List[Dict[str, Any]]:
        """Check if system prompt has leaked into response"""
        issues = []
        
        for pattern in self._prompt_leak_patterns:
            if pattern.search(response):
                issues.append({
                    'type': ValidationIssue.PROMPT_LEAK,
                    'severity': ValidationSeverity.HIGH,
                    'description': f"Possible system prompt leak detected: {pattern.pattern[:30]}...",
                    'location': pattern.pattern
                })
        
        return issues
    
    def _check_template_response(self, response: str) -> List[Dict[str, Any]]:
        """Check for generic template-like responses"""
        issues = []
        response_lower = response.lower()
        
        template_matches = []
        for indicator in self.TEMPLATE_INDICATORS:
            if indicator.lower() in response_lower:
                template_matches.append(indicator)
        
        if len(template_matches) >= 2:
            issues.append({
                'type': ValidationIssue.TEMPLATE_RESPONSE,
                'severity': ValidationSeverity.MEDIUM,
                'description': f"Response contains {len(template_matches)} template phrases",
                'matches': template_matches
            })
        
        # Check for very short or evasive responses
        if len(response.strip()) < 50 and "sorry" in response_lower:
            issues.append({
                'type': ValidationIssue.TEMPLATE_RESPONSE,
                'severity': ValidationSeverity.MEDIUM,
                'description': "Short evasive response detected"
            })
        
        return issues
    
    def _check_language(self, response: str, expected_lang: str) -> List[Dict[str, Any]]:
        """Check if response is in expected language"""
        issues = []
        
        # Simple heuristics for language detection
        # Turkish indicators
        turkish_chars = set('ğüşıöçĞÜŞİÖÇ')
        has_turkish = any(c in response for c in turkish_chars)
        
        # Russian indicators (Cyrillic)
        russian_range = range(0x0400, 0x04FF)
        has_cyrillic = any(ord(c) in russian_range for c in response)
        
        # Arabic indicators
        arabic_range = range(0x0600, 0x06FF)
        has_arabic = any(ord(c) in arabic_range for c in response)
        
        # German indicators
        german_words = ['und', 'der', 'die', 'das', 'ist', 'sie', 'nicht']
        has_german = any(f' {w} ' in f' {response.lower()} ' for w in german_words)
        
        # Check consistency
        if expected_lang == 'en':
            if has_turkish or has_cyrillic or has_arabic:
                issues.append({
                    'type': ValidationIssue.WRONG_LANGUAGE,
                    'severity': ValidationSeverity.LOW,
                    'description': f"Response may contain non-English content"
                })
        elif expected_lang == 'tr':
            if has_cyrillic or has_arabic:
                issues.append({
                    'type': ValidationIssue.WRONG_LANGUAGE,
                    'severity': ValidationSeverity.MEDIUM,
                    'description': "Turkish response contains other script"
                })
        elif expected_lang == 'ru':
            if not has_cyrillic:
                issues.append({
                    'type': ValidationIssue.WRONG_LANGUAGE,
                    'severity': ValidationSeverity.MEDIUM,
                    'description': "Russian response should use Cyrillic script"
                })
        
        return issues
    
    def _check_route_consistency(
        self,
        response: str,
        route_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check if response is consistent with verified route data"""
        issues = []
        response_lower = response.lower()
        
        # Check if stated duration matches
        if route_data.get('total_time'):
            correct_time = str(route_data['total_time'])
            
            # Look for time mentions that don't match
            time_pattern = r'(\d+)\s*(?:minutes?|mins?|dakika)'
            time_matches = re.findall(time_pattern, response_lower)
            
            for match in time_matches:
                if abs(int(match) - int(correct_time)) > 5:  # Allow 5 min variance
                    issues.append({
                        'type': ValidationIssue.CONTRADICTION,
                        'severity': ValidationSeverity.HIGH,
                        'description': f"Route duration mismatch: stated {match} min, verified {correct_time} min"
                    })
                    break
        
        # Check if transit lines mentioned are correct
        if route_data.get('lines_used'):
            correct_lines = set(route_data['lines_used'])
            
            # Common line patterns
            line_pattern = r'\b(M\d+|T\d+|F\d+|Marmaray)\b'
            mentioned_lines = set(re.findall(line_pattern, response, re.IGNORECASE))
            
            # Check for lines not in route
            wrong_lines = mentioned_lines - correct_lines
            if wrong_lines and len(mentioned_lines) > 1:  # Allow some flexibility
                issues.append({
                    'type': ValidationIssue.HALLUCINATION,
                    'severity': ValidationSeverity.MEDIUM,
                    'description': f"Mentioned transit lines not in verified route: {wrong_lines}"
                })
        
        # Check transfer count
        if 'transfers' in route_data:
            correct_transfers = route_data['transfers']
            
            # Look for transfer mentions
            transfer_words = ['transfer', 'change', 'aktarma', 'değiştir']
            transfer_count = sum(1 for w in transfer_words if w in response_lower)
            
            # This is a weak check - just flag if significantly off
            if transfer_count > correct_transfers + 2:
                issues.append({
                    'type': ValidationIssue.HALLUCINATION,
                    'severity': ValidationSeverity.LOW,
                    'description': f"Response mentions more transfers than verified: {transfer_count} vs {correct_transfers}"
                })
        
        return issues
    
    def _check_hallucinations(
        self,
        response: str,
        context_sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check for potential hallucinations.
        
        This is a heuristic check - looks for specific claims
        that don't appear in any context source.
        """
        issues = []
        
        # Build context text for checking
        all_context = " ".join([
            src.get('content', '') for src in context_sources
        ]).lower()
        
        # Check for specific time claims
        time_pattern = r'(?:opens?|closes?|hours?)\s*(?:at|:)?\s*(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?)'
        time_claims = re.findall(time_pattern, response.lower())
        
        for claim in time_claims:
            if claim not in all_context:
                issues.append({
                    'type': ValidationIssue.HALLUCINATION,
                    'severity': ValidationSeverity.LOW,
                    'description': f"Time claim '{claim}' not found in context"
                })
        
        # Check for price claims
        price_pattern = r'(?:costs?|prices?|fee|₺|TL)\s*(?:is|:)?\s*([\d,.]+)'
        price_claims = re.findall(price_pattern, response.lower())
        
        for claim in price_claims:
            if claim not in all_context:
                issues.append({
                    'type': ValidationIssue.HALLUCINATION,
                    'severity': ValidationSeverity.LOW,
                    'description': f"Price claim '{claim}' not verified in context"
                })
        
        # Check for phone number claims
        phone_pattern = r'(?:phone|call|tel)\s*(?::|is)?\s*([\d\s\-+()]{7,})'
        phone_claims = re.findall(phone_pattern, response.lower())
        
        for claim in phone_claims:
            clean_claim = re.sub(r'[\s\-()]', '', claim)
            if clean_claim not in re.sub(r'[\s\-()]', '', all_context):
                issues.append({
                    'type': ValidationIssue.HALLUCINATION,
                    'severity': ValidationSeverity.MEDIUM,
                    'description': f"Phone number '{claim}' not verified"
                })
        
        return issues
    
    def _check_over_confidence(
        self,
        response: str,
        context_sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check for over-confident claims"""
        issues = []
        response_lower = response.lower()
        
        # Count over-confident phrases
        confident_count = sum(
            1 for phrase in self.OVER_CONFIDENT_PHRASES
            if phrase.lower() in response_lower
        )
        
        # Check context confidence
        avg_confidence = 1.0
        if context_sources:
            confidences = [
                src.get('confidence', 1.0) for src in context_sources
            ]
            avg_confidence = sum(confidences) / len(confidences)
        
        # Flag if very confident language with low-confidence context
        if confident_count >= 2 and avg_confidence < 0.7:
            issues.append({
                'type': ValidationIssue.OVER_CONFIDENT,
                'severity': ValidationSeverity.LOW,
                'description': f"Response uses {confident_count} confident phrases with {avg_confidence:.0%} context confidence"
            })
        
        return issues
    
    def _determine_action(self, issues: List[Dict[str, Any]]) -> str:
        """Determine what action to take based on issues"""
        if not issues:
            return "none"
        
        severity_counts = {
            ValidationSeverity.LOW: 0,
            ValidationSeverity.MEDIUM: 0,
            ValidationSeverity.HIGH: 0,
            ValidationSeverity.CRITICAL: 0
        }
        
        for issue in issues:
            severity = issue.get('severity', ValidationSeverity.LOW)
            severity_counts[severity] += 1
        
        if severity_counts[ValidationSeverity.CRITICAL] > 0:
            return "abort"
        elif severity_counts[ValidationSeverity.HIGH] >= 2:
            return "repair"
        elif severity_counts[ValidationSeverity.HIGH] == 1:
            return "hedge"
        elif severity_counts[ValidationSeverity.MEDIUM] >= 3:
            return "hedge"
        else:
            return "none"
    
    def _generate_hedging(self, language: str) -> List[str]:
        """Generate hedging phrases for the language"""
        return self.HEDGING_PHRASES.get(language, self.HEDGING_PHRASES['en'])
    
    def apply_hedging(
        self,
        response: str,
        hedging_phrases: List[str],
        language: str = "en"
    ) -> str:
        """
        Apply hedging to a response.
        
        Adds uncertainty markers where appropriate.
        """
        if not hedging_phrases:
            return response
        
        # Simple approach: prepend a hedging phrase if response starts with a fact
        # More sophisticated: identify specific claims and add hedging
        
        # Check if response starts with a confident claim
        starts_confident = any(
            response.lower().startswith(phrase.lower())
            for phrase in ["the", "it is", "you can", "there are", "this is"]
        )
        
        if starts_confident:
            import random
            hedge = random.choice(hedging_phrases)
            return hedge + response[0].lower() + response[1:]
        
        return response
    
    def check_streaming_quality(
        self,
        partial_response: str,
        min_length: int = 50
    ) -> Tuple[bool, Optional[str]]:
        """
        Check partial response quality during streaming.
        
        Returns (should_continue, reason_to_abort)
        """
        # Check if it's just starting
        if len(partial_response) < min_length:
            return True, None
        
        # Check for prompt leak early
        for pattern in self._prompt_leak_patterns[:3]:  # Check first few patterns only
            if pattern.search(partial_response):
                return False, "Possible prompt leak detected"
        
        # Check for template start
        template_starts = [
            "I'd be happy to help",
            "I apologize, but",
            "I'm sorry, but I",
            "As an AI",
            "I don't have access",
        ]
        
        for starter in template_starts:
            if partial_response.strip().startswith(starter):
                return False, f"Template response detected: {starter[:20]}..."
        
        return True, None


# Global instance
_validator: Optional[ResponseValidator] = None


def get_response_validator() -> ResponseValidator:
    """Get or create global response validator instance"""
    global _validator
    if _validator is None:
        _validator = ResponseValidator()
    return _validator
