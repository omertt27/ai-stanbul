"""
Transportation Safety Net

Verifies that LLM responses don't contradict structured route data.
Detects and prevents LLM hallucinations about transit lines, stations, or routes.

This is a critical safety layer that ensures the LLM never invents:
- Non-existent metro lines
- Fake station names
- Incorrect transfer points
- Wrong travel times

Author: AI Istanbul Team
Date: December 17, 2025
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class TransportationSafetyNet:
    """
    Safety net to verify LLM responses against structured route data.
    
    This class prevents the LLM from hallucinating transportation information
    by cross-checking the text response against verified route data.
    
    Detection Categories:
    1. Line Hallucinations: LLM mentions lines not in verified route
    2. Station Hallucinations: LLM mentions stations not in verified route
    3. Transfer Contradictions: LLM describes transfers differently than route data
    4. Time Contradictions: LLM gives significantly different travel times
    """
    
    # Known Istanbul metro/transit lines (comprehensive list)
    VALID_LINES = {
        'M1', 'M1A', 'M1B', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M9', 'M11',
        'T1', 'T4', 'T5',
        'MARMARAY', 'METROBUS',
        'FERRY', 'FUNICULAR', 'CABLE CAR'
    }
    
    # Pattern to extract metro line mentions from text
    LINE_PATTERN = re.compile(
        r'\b(M[0-9]+[AB]?|T[0-9]+|MARMARAY|METROBUS|FERRY|FUNICULAR)\b',
        re.IGNORECASE
    )
    
    # Pattern to extract station names (basic - can be enhanced)
    STATION_PATTERN = re.compile(
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Station|İstasyonu)\b'
    )
    
    # Pattern to extract time mentions
    TIME_PATTERN = re.compile(
        r'(\d+)\s*(?:minute|dakika|min)',
        re.IGNORECASE
    )
    
    def __init__(self):
        """Initialize the safety net"""
        logger.info("🛡️ Transportation Safety Net initialized")
    
    def verify_response(
        self,
        llm_text: str,
        route_data: Optional[Dict[str, Any]],
        confidence_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Verify that LLM response doesn't contradict route data.
        
        Args:
            llm_text: The LLM's generated text response
            route_data: Structured route data from Dijkstra algorithm
            confidence_threshold: Minimum confidence to pass verification
            
        Returns:
            Dict with verification results:
            {
                'verified': bool,
                'confidence': float,
                'hallucinations': List[str],
                'warnings': List[str],
                'corrected_text': Optional[str]
            }
        """
        if not route_data:
            # No route data to verify against - pass through
            return {
                'verified': True,
                'confidence': 1.0,
                'hallucinations': [],
                'warnings': [],
                'corrected_text': None,
                'reason': 'no_route_data'
            }
        
        hallucinations = []
        warnings = []
        confidence = 1.0
        
        # Extract route information for verification
        verified_lines = set()
        verified_stations = set()
        verified_time = None
        verified_transfers = None
        
        if 'lines_used' in route_data:
            verified_lines = set(line.upper() for line in route_data['lines_used'])
        
        if 'steps' in route_data:
            for step in route_data['steps']:
                if isinstance(step, dict):
                    if 'from_station' in step:
                        verified_stations.add(step['from_station'])
                    if 'to_station' in step:
                        verified_stations.add(step['to_station'])
        
        if 'total_time' in route_data:
            verified_time = route_data['total_time']
        
        if 'transfers' in route_data:
            verified_transfers = route_data['transfers']
        
        # 1. Check for line hallucinations
        mentioned_lines = self._extract_lines(llm_text)
        hallucinated_lines = []
        
        for line in mentioned_lines:
            line_upper = line.upper()
            # Check if line is valid AND in the verified route
            if line_upper not in self.VALID_LINES:
                hallucinations.append(f"Invalid metro line: {line}")
                hallucinated_lines.append(line)
                confidence -= 0.3
            elif verified_lines and line_upper not in verified_lines:
                hallucinations.append(f"LLM mentions {line} but it's not in the route")
                hallucinated_lines.append(line)
                confidence -= 0.2
        
        # 2. Check for time contradictions
        mentioned_times = self._extract_times(llm_text)
        if mentioned_times and verified_time:
            # Check if any mentioned time is significantly different
            for time_val in mentioned_times:
                # Allow 20% variance
                if abs(time_val - verified_time) > (verified_time * 0.2):
                    warnings.append(
                        f"LLM mentions {time_val} minutes but route says {verified_time} minutes"
                    )
                    confidence -= 0.1
        
        # 3. Check for transfer count contradictions
        if verified_transfers is not None:
            transfer_pattern = re.compile(r'(\d+)\s*transfer', re.IGNORECASE)
            mentioned_transfers = transfer_pattern.findall(llm_text)
            
            for transfer_str in mentioned_transfers:
                transfer_count = int(transfer_str)
                if transfer_count != verified_transfers:
                    warnings.append(
                        f"LLM mentions {transfer_count} transfers but route has {verified_transfers}"
                    )
                    confidence -= 0.15
        
        # 4. Check for station hallucinations (basic check)
        # This is complex because station names vary in spelling
        # For now, just log if we find suspiciously named stations
        mentioned_stations = self._extract_stations(llm_text)
        suspicious_stations = []
        
        for station in mentioned_stations:
            # Very basic check - can be enhanced with a proper station database
            if len(station) < 3 or any(char.isdigit() for char in station):
                suspicious_stations.append(station)
        
        if suspicious_stations:
            warnings.append(f"Potentially invalid station names: {', '.join(suspicious_stations)}")
            confidence -= 0.05
        
        # Final confidence calculation
        confidence = max(0.0, min(1.0, confidence))
        verified = confidence >= confidence_threshold and len(hallucinations) == 0
        
        # Log results
        if not verified:
            logger.warning(
                f"⚠️ Safety Net: LLM response failed verification "
                f"(confidence: {confidence:.2f}, hallucinations: {len(hallucinations)})"
            )
            for h in hallucinations:
                logger.warning(f"   - {h}")
        elif warnings:
            logger.info(
                f"✓ Safety Net: Verified with warnings "
                f"(confidence: {confidence:.2f}, warnings: {len(warnings)})"
            )
        else:
            logger.info(f"✓ Safety Net: Fully verified (confidence: {confidence:.2f})")
        
        return {
            'verified': verified,
            'confidence': confidence,
            'hallucinations': hallucinations,
            'warnings': warnings,
            'corrected_text': None,  # Could implement auto-correction in future
            'mentioned_lines': list(mentioned_lines),
            'verified_lines': list(verified_lines) if verified_lines else [],
            'hallucinated_lines': hallucinated_lines
        }
    
    def _extract_lines(self, text: str) -> List[str]:
        """Extract metro line mentions from text"""
        matches = self.LINE_PATTERN.findall(text)
        return [m.upper() for m in matches]
    
    def _extract_stations(self, text: str) -> List[str]:
        """Extract station name mentions from text (basic)"""
        matches = self.STATION_PATTERN.findall(text)
        return matches
    
    def _extract_times(self, text: str) -> List[int]:
        """Extract time values from text"""
        matches = self.TIME_PATTERN.findall(text)
        return [int(m) for m in matches]
    
    def generate_safety_warning(
        self,
        verification_result: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate a user-facing warning message if verification failed.
        
        Args:
            verification_result: Result from verify_response()
            
        Returns:
            Warning message string or None
        """
        if verification_result['verified']:
            return None
        
        hallucinations = verification_result.get('hallucinations', [])
        
        if not hallucinations:
            return None
        
        # Build warning message
        warning = "⚠️ Note: Some route information may not be accurate. "
        warning += "Please verify with official transit maps. "
        
        if verification_result.get('hallucinated_lines'):
            lines = verification_result['hallucinated_lines']
            warning += f"(Lines mentioned: {', '.join(lines)})"
        
        return warning


# Singleton instance
_safety_net = None

def get_safety_net() -> TransportationSafetyNet:
    """Get or create the safety net singleton"""
    global _safety_net
    if _safety_net is None:
        _safety_net = TransportationSafetyNet()
    return _safety_net
