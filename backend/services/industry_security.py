"""
Industry-Level Input Validation and Security System
=================================================

Enterprise-grade input validation, sanitization, and security controls
for the AI Istanbul system. Implements defense-in-depth security.
"""

import re
import json
import hashlib
import time
import logging
from typing import Dict, List, Optional, Any, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import html
import urllib.parse
import ipaddress
import uuid
from functools import wraps

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ValidationResult(Enum):
    """Validation result types"""
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"
    SANITIZE = "sanitize"

@dataclass
class SecurityEvent:
    """Security event structure"""
    event_id: str
    event_type: str
    severity: SecurityLevel
    source_ip: Optional[str]
    user_agent: Optional[str]
    payload: Dict[str, Any]
    timestamp: datetime
    blocked: bool
    sanitized: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationRule:
    """Input validation rule"""
    name: str
    pattern: str
    severity: SecurityLevel
    action: ValidationResult
    description: str
    enabled: bool = True

class IndustrySecuritySystem:
    """
    Enterprise security system with:
    - Multi-layer input validation
    - Advanced threat detection
    - Real-time security monitoring
    - Automated response mechanisms
    - Compliance controls
    """
    
    def __init__(self):
        # Security patterns
        self.security_patterns = self._load_security_patterns()
        self.validation_rules = self._load_validation_rules()
        
        # Rate limiting
        self.rate_limits = {}
        self.blocked_ips = set()
        self.suspicious_ips = {}
        
        # Security events storage
        self.security_events = []
        self.threat_intelligence = {}
        
        # Sanitization settings
        self.max_input_length = 10000
        self.allowed_html_tags = set()  # No HTML allowed by default
        self.allowed_protocols = {'http', 'https'}
        
        # Initialize monitoring
        try:
            from .industry_monitoring import get_monitoring_system
            self.monitoring = get_monitoring_system()
            self.monitoring_enabled = True
        except ImportError:
            self.monitoring = None
            self.monitoring_enabled = False
        
        logger.info("🛡️ Industry Security System initialized")
    
    def _load_security_patterns(self) -> Dict[str, List[str]]:
        """Load comprehensive security threat patterns"""
        return {
            "sql_injection": [
                # Classic SQL injection patterns
                r"(?i)(union\s+select|select\s+\*\s+from|insert\s+into|delete\s+from|drop\s+table)",
                r"(?i)(\'\s*or\s+\'1\'\s*=\s*\'1|--\s*$|\*\s*\))",
                r"(?i)(exec\s*\(|execute\s*\(|sp_|xp_)",
                r"(?i)(information_schema|sys\.tables|mysql\.user)",
                r"(?i)(\bor\b\s+\d+\s*=\s*\d+|\band\b\s+\d+\s*=\s*\d+)",
                
                # Advanced SQL injection
                r"(?i)(waitfor\s+delay|benchmark\s*\(|sleep\s*\()",
                r"(?i)(load_file\s*\(|into\s+outfile|into\s+dumpfile)",
                r"(?i)(char\s*\(\d+\)|concat\s*\(|substring\s*\()",
                r"(?i)(\|\|\s*\'|\'\s*\|\||concat\s*\()",
                r"(?i)(0x[0-9a-f]+|ascii\s*\(|hex\s*\()"
            ],
            
            "xss_injection": [
                # Basic XSS patterns
                r"(?i)(<script[^>]*>|</script>|javascript:|vbscript:)",
                r"(?i)(on\w+\s*=|<iframe|<object|<embed)",
                r"(?i)(<img[^>]+src\s*=\s*[\"']?javascript:|<img[^>]+onerror)",
                r"(?i)(alert\s*\(|confirm\s*\(|prompt\s*\()",
                r"(?i)(document\.(cookie|domain|location)|window\.(location|open))",
                
                # Advanced XSS patterns
                r"(?i)(expression\s*\(|@import|behavior\s*:)",
                r"(?i)(&#x?[0-9a-f]+;|%[0-9a-f][0-9a-f])",
                r"(?i)(eval\s*\(|settimeout\s*\(|setinterval\s*\()",
                r"(?i)(<svg[^>]*on\w+|<math[^>]*on\w+)",
                r"(?i)(data:text/html|data:image/svg)"
            ],
            
            "command_injection": [
                # System command patterns
                r"(?i)(;|\||\&\&|\|\|).*?(cat|ls|pwd|whoami|id|uname)",
                r"(?i)(wget|curl|nc|netcat|telnet|ssh)",
                r"(?i)(`[^`]*`|\$\([^)]*\))",
                r"(?i)(rm\s+-rf|mkfs|fdisk|format)",
                r"(?i)(python|perl|php|ruby|bash|sh)\s+",
                
                # Windows command injection
                r"(?i)(cmd\.exe|powershell|net\s+user|reg\s+add)",
                r"(?i)(type\s+|copy\s+|move\s+|del\s+|rd\s+)",
                r"(?i)(ping\s+-[tn]|nslookup|arp\s+)",
                r"(?i)(tasklist|taskkill|sc\s+query)"
            ],
            
            "path_traversal": [
                # Directory traversal patterns
                r"(?i)(\.\.[\\/]|[\\/]\.\.)",
                r"(?i)(\.\.%2f|%2f\.\.|\.\.%5c|%5c\.\.)",
                r"(?i)(%252e%252e|%c0%ae%c0%ae)",
                r"(?i)([\\/]etc[\\/]passwd|[\\/]windows[\\/]system32)",
                r"(?i)(proc[\\/]self[\\/]|dev[\\/]|var[\\/]log)"
            ],
            
            "ldap_injection": [
                # LDAP injection patterns
                r"(?i)(\*\)\(|\)\(\*|\*\)\(\&)",
                r"(?i)(\)\(\|.*?\=|\&\(.*?\=)",
                r"(?i)(\(cn\=\*\)|\(uid\=\*\)|\(mail\=\*\))",
                r"(?i)(\|\(|\)\&|\&\(|\)\|)"
            ],
            
            "template_injection": [
                # Template injection patterns
                r"(?i)(\{\{.*?\}\}|\{%.*?%\}|\${.*?})",
                r"(?i)(\[\[.*?\]\]|<%.*?%>|<\?.*?\?>)",
                r"(?i)(config\.|request\.|session\.)",
                r"(?i)(__import__|getattr|setattr|delattr)",
                r"(?i)(self\.__class__|mro\(\)|subclasses\(\))"
            ],
            
            "nosql_injection": [
                # NoSQL injection patterns
                r"(?i)(\$ne\s*:|ne\s*:|regex\s*:|\$regex\s*:)",
                r"(?i)(\$where\s*:|\$gt\s*:|\$lt\s*:|\$gte\s*:)",
                r"(?i)(\$in\s*:|\$nin\s*:|\$or\s*:|\$and\s*:)",
                r"(?i)(javascript\s*:|function\s*\(|this\.)",
                r"(?i)(\$exists\s*:|\$type\s*:|\$size\s*:)"
            ],
            
            "deserialization": [
                # Deserialization attack patterns
                r"(?i)(pickle\.loads|marshal\.loads|yaml\.load)",
                r"(?i)(__reduce__|__setstate__|__getstate__)",
                r"(?i)(rO0AB|aced0005|H4sIA)",  # Java/Python serialization signatures
                r"(?i)(ObjectInputStream|readObject|writeObject)",
                r"(?i)(subprocess\.call|os\.system|eval\()"
            ]
        }
    
    def _load_validation_rules(self) -> List[ValidationRule]:
        """Load input validation rules"""
        return [
            # Length validation
            ValidationRule(
                "max_length",
                f".{{{self.max_input_length + 1},}}",
                SecurityLevel.MEDIUM,
                ValidationResult.BLOCK,
                f"Input exceeds maximum length of {self.max_input_length} characters"
            ),
            
            # Null byte injection
            ValidationRule(
                "null_bytes",
                r"[\x00]",
                SecurityLevel.HIGH,
                ValidationResult.BLOCK,
                "Null byte injection attempt"
            ),
            
            # Control characters
            ValidationRule(
                "control_chars",
                r"[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]",
                SecurityLevel.MEDIUM,
                ValidationResult.SANITIZE,
                "Potentially malicious control characters"
            ),
            
            # Unicode homograph attacks
            ValidationRule(
                "homograph_attack",
                r"[а-я].*[a-z]|[a-z].*[а-я]",  # Mixed Cyrillic and Latin
                SecurityLevel.MEDIUM,
                ValidationResult.WARN,
                "Potential homograph attack (mixed character sets)"
            ),
            
            # Excessive special characters
            ValidationRule(
                "excessive_special_chars",
                r"[!@#$%^&*()_+=\[\]{};':\"\\|,.<>?/~`-]{50,}",
                SecurityLevel.MEDIUM,
                ValidationResult.WARN,
                "Excessive special characters detected"
            ),
            
            # Base64 encoded payloads
            ValidationRule(
                "base64_payload",
                r"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?",
                SecurityLevel.LOW,
                ValidationResult.WARN,
                "Potential base64 encoded payload"
            ),
            
            # URL encoding attacks
            ValidationRule(
                "url_encoding_attack",
                r"(%[0-9a-fA-F]{2}){10,}",
                SecurityLevel.MEDIUM,
                ValidationResult.WARN,
                "Excessive URL encoding detected"
            )
        ]
    
    def validate_input(self, input_data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive input validation with threat detection
        
        Args:
            input_data: The input to validate
            context: Additional context (IP, user agent, etc.)
            
        Returns:
            Validation result with security assessment
        """
        if context is None:
            context = {}
        
        # Convert input to string for analysis
        if isinstance(input_data, dict):
            input_str = json.dumps(input_data, sort_keys=True)
        elif isinstance(input_data, (list, tuple)):
            input_str = str(input_data)
        else:
            input_str = str(input_data)
        
        # Initialize result
        result = {
            "valid": True,
            "sanitized": False,
            "blocked": False,
            "warnings": [],
            "threats": [],
            "sanitized_input": input_data,
            "security_score": 100,
            "metadata": {}
        }
        
        # Run validation rules
        for rule in self.validation_rules:
            if not rule.enabled:
                continue
                
            if re.search(rule.pattern, input_str, re.IGNORECASE | re.MULTILINE):
                threat = {
                    "rule": rule.name,
                    "severity": rule.severity.value,
                    "action": rule.action.value,
                    "description": rule.description
                }
                
                result["threats"].append(threat)
                result["security_score"] -= self._get_severity_penalty(rule.severity)
                
                if rule.action == ValidationResult.BLOCK:
                    result["valid"] = False
                    result["blocked"] = True
                elif rule.action == ValidationResult.WARN:
                    result["warnings"].append(rule.description)
                elif rule.action == ValidationResult.SANITIZE:
                    result["sanitized"] = True
                    result["sanitized_input"] = self._sanitize_input(input_str, rule.pattern)
        
        # Run threat pattern detection
        for threat_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_str, re.IGNORECASE | re.MULTILINE):
                    threat = {
                        "type": threat_type,
                        "pattern": pattern[:50] + "...",
                        "severity": "high",
                        "confidence": self._calculate_threat_confidence(pattern, input_str)
                    }
                    
                    result["threats"].append(threat)
                    result["security_score"] -= 25
                    
                    # Block high-confidence threats
                    if threat["confidence"] > 0.8:
                        result["valid"] = False
                        result["blocked"] = True
        
        # Log security events
        if result["threats"] or result["security_score"] < 80:
            self._log_security_event(input_str, context, result)
        
        # Rate limiting check
        source_ip = context.get("source_ip")
        if source_ip:
            if not self._check_rate_limit(source_ip):
                result["valid"] = False
                result["blocked"] = True
                result["threats"].append({
                    "type": "rate_limit_exceeded",
                    "severity": "medium",
                    "description": "Rate limit exceeded"
                })
        
        return result
    
    def sanitize_input(self, input_data: Any, aggressive: bool = False) -> Any:
        """
        Sanitize input data with configurable aggressiveness
        
        Args:
            input_data: Input to sanitize
            aggressive: Whether to apply aggressive sanitization
            
        Returns:
            Sanitized input
        """
        if isinstance(input_data, str):
            return self._sanitize_string(input_data, aggressive)
        elif isinstance(input_data, dict):
            return {k: self.sanitize_input(v, aggressive) for k, v in input_data.items()}
        elif isinstance(input_data, list):
            return [self.sanitize_input(item, aggressive) for item in input_data]
        else:
            return input_data
    
    def _sanitize_string(self, text: str, aggressive: bool = False) -> str:
        """Sanitize a string input"""
        if not isinstance(text, str):
            return text
        
        # Basic sanitization
        sanitized = text
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Remove control characters
        sanitized = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        # HTML escape
        sanitized = html.escape(sanitized, quote=True)
        
        # URL decode (but not recursively to avoid double-decoding attacks)
        try:
            decoded = urllib.parse.unquote(sanitized)
            if decoded != sanitized:  # Only use if different
                sanitized = html.escape(decoded, quote=True)
        except:
            pass  # Keep original if decoding fails
        
        if aggressive:
            # More aggressive sanitization
            # Remove potentially dangerous patterns
            dangerous_patterns = [
                r'<[^>]*>',  # HTML tags
                r'javascript:',  # JavaScript URLs
                r'vbscript:',   # VBScript URLs
                r'data:',       # Data URLs
                r'eval\s*\(',   # eval() calls
                r'expression\s*\(',  # CSS expressions
            ]
            
            for pattern in dangerous_patterns:
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
            
            # Limit special characters
            sanitized = re.sub(r'[!@#$%^&*()_+=\[\]{};\':"\\|,.<>?/~`-]{10,}', 
                             lambda m: m.group(0)[:10], sanitized)
        
        # Truncate if too long
        if len(sanitized) > self.max_input_length:
            sanitized = sanitized[:self.max_input_length]
        
        return sanitized
    
    def _sanitize_input(self, input_str: str, pattern: str) -> str:
        """Sanitize input based on specific pattern"""
        # Remove matches of the dangerous pattern
        return re.sub(pattern, '', input_str, flags=re.IGNORECASE | re.MULTILINE)
    
    def _calculate_threat_confidence(self, pattern: str, input_str: str) -> float:
        """Calculate confidence level for threat detection"""
        matches = re.findall(pattern, input_str, re.IGNORECASE | re.MULTILINE)
        
        if not matches:
            return 0.0
        
        # Base confidence on number of matches and pattern complexity
        base_confidence = min(len(matches) * 0.2, 0.8)
        
        # Increase confidence for exact matches of dangerous patterns
        dangerous_keywords = ['script', 'eval', 'exec', 'union', 'select', 'drop', 'delete']
        for keyword in dangerous_keywords:
            if keyword.lower() in input_str.lower():
                base_confidence += 0.2
        
        return min(base_confidence, 1.0)
    
    def _get_severity_penalty(self, severity: SecurityLevel) -> int:
        """Get security score penalty based on severity"""
        penalties = {
            SecurityLevel.LOW: 5,
            SecurityLevel.MEDIUM: 15,
            SecurityLevel.HIGH: 30,
            SecurityLevel.CRITICAL: 50
        }
        return penalties.get(severity, 10)
    
    def _check_rate_limit(self, source_ip: str) -> bool:
        """Check if source IP exceeds rate limits"""
        current_time = time.time()
        
        # Clean old entries
        if source_ip in self.rate_limits:
            self.rate_limits[source_ip] = [
                timestamp for timestamp in self.rate_limits[source_ip]
                if current_time - timestamp < 3600  # 1 hour window
            ]
        else:
            self.rate_limits[source_ip] = []
        
        # Check limits
        requests_last_hour = len(self.rate_limits[source_ip])
        requests_last_minute = len([
            t for t in self.rate_limits[source_ip]
            if current_time - t < 60
        ])
        
        # Rate limits: 100 requests per hour, 10 per minute
        if requests_last_hour >= 100 or requests_last_minute >= 10:
            self.blocked_ips.add(source_ip)
            return False
        
        # Record this request
        self.rate_limits[source_ip].append(current_time)
        return True
    
    def _log_security_event(self, input_str: str, context: Dict[str, Any], 
                           validation_result: Dict[str, Any]):
        """Log security event for monitoring"""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type="input_validation",
            severity=SecurityLevel.MEDIUM if validation_result["blocked"] else SecurityLevel.LOW,
            source_ip=context.get("source_ip"),
            user_agent=context.get("user_agent"),
            payload={
                "input_preview": input_str[:200] + "..." if len(input_str) > 200 else input_str,
                "input_length": len(input_str),
                "threats_detected": len(validation_result["threats"]),
                "security_score": validation_result["security_score"],
                "blocked": validation_result["blocked"]
            },
            timestamp=datetime.now(),
            blocked=validation_result["blocked"],
            sanitized=validation_result["sanitized"]
        )
        
        self.security_events.append(event)
        
        # Send to monitoring system
        if self.monitoring_enabled and self.monitoring:
            self.monitoring.log_security_event(
                event.event_type,
                event.severity,
                event.source_ip,
                event.user_agent,
                event.payload,
                event.blocked
            )
        
        # Log to standard logger
        level = logging.WARNING if event.blocked else logging.INFO
        logger.log(level, f"🛡️ Security Event: {event.event_type} from {event.source_ip} "
                         f"[Score: {validation_result['security_score']}, "
                         f"Threats: {len(validation_result['threats'])}, "
                         f"Blocked: {event.blocked}]")
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        return ip in self.blocked_ips
    
    def block_ip(self, ip: str, reason: str = "Security violation"):
        """Block an IP address"""
        self.blocked_ips.add(ip)
        
        if self.monitoring_enabled and self.monitoring:
            self.monitoring.log_security_event(
                "ip_blocked",
                SecurityLevel.HIGH,
                ip,
                None,
                {"reason": reason},
                True
            )
        
        logger.warning(f"🚫 IP {ip} blocked: {reason}")
    
    def unblock_ip(self, ip: str):
        """Unblock an IP address"""
        self.blocked_ips.discard(ip)
        logger.info(f"✅ IP {ip} unblocked")
    
    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive security report"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.security_events if e.timestamp > cutoff_time]
        
        # Count events by type and severity
        event_types = {}
        severity_counts = {level.value: 0 for level in SecurityLevel}
        blocked_count = 0
        
        for event in recent_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            severity_counts[event.severity.value] += 1
            if event.blocked:
                blocked_count += 1
        
        # Top threat sources
        ip_threats = {}
        for event in recent_events:
            if event.source_ip:
                ip_threats[event.source_ip] = ip_threats.get(event.source_ip, 0) + 1
        
        top_threat_ips = sorted(ip_threats.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "time_period_hours": hours,
            "total_events": len(recent_events),
            "blocked_events": blocked_count,
            "blocked_percentage": (blocked_count / len(recent_events) * 100) if recent_events else 0,
            "event_types": event_types,
            "severity_distribution": severity_counts,
            "top_threat_ips": top_threat_ips,
            "currently_blocked_ips": len(self.blocked_ips),
            "rate_limited_ips": len(self.rate_limits),
            "threat_patterns_detected": len([e for e in recent_events if e.payload.get("threats_detected", 0) > 0])
        }
    
    def update_threat_intelligence(self, intelligence_data: Dict[str, Any]):
        """Update threat intelligence database"""
        self.threat_intelligence.update(intelligence_data)
        logger.info(f"🔄 Threat intelligence updated with {len(intelligence_data)} entries")
    
    def export_security_logs(self, format: str = "json") -> str:
        """Export security logs in specified format"""
        if format.lower() == "json":
            return json.dumps([
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "severity": event.severity.value,
                    "source_ip": event.source_ip,
                    "timestamp": event.timestamp.isoformat(),
                    "blocked": event.blocked,
                    "payload": event.payload
                }
                for event in self.security_events
            ], indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Global security instance
_security_instance = None

def get_security_system() -> IndustrySecuritySystem:
    """Get the global security system instance"""
    global _security_instance
    if _security_instance is None:
        _security_instance = IndustrySecuritySystem()
    return _security_instance

def secure_endpoint(require_validation: bool = True, rate_limit: bool = True):
    """Decorator to secure API endpoints with validation and rate limiting"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            security = get_security_system()
            
            # Extract request context (this would need to be adapted for your framework)
            context = {
                "source_ip": getattr(kwargs.get('request'), 'client_ip', 'unknown'),
                "user_agent": getattr(kwargs.get('request'), 'user_agent', 'unknown')
            }
            
            # Check if IP is blocked
            if context["source_ip"] != 'unknown' and security.is_ip_blocked(context["source_ip"]):
                raise PermissionError("IP address is blocked")
            
            # Validate input data if required
            if require_validation:
                for arg in args:
                    if isinstance(arg, (str, dict, list)):
                        validation_result = security.validate_input(arg, context)
                        if not validation_result["valid"]:
                            raise ValueError("Input validation failed")
            
            # Call original function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def validate_and_sanitize(input_data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function for input validation and sanitization"""
    security = get_security_system()
    return security.validate_input(input_data, context)

# Initialize security system when module is imported
if __name__ != "__main__":
    try:
        _security_instance = IndustrySecuritySystem()
        logger.info("🛡️ Industry Security System auto-initialized")
    except Exception as e:
        logger.error(f"❌ Failed to auto-initialize security system: {e}")
