"""
Input sanitization middleware for the Istanbul AI chatbot API.
Provides XSS protection, SQL injection prevention, and input validation.
"""

import re
import html
import json
from typing import Dict, Any, List, Optional, Union
from urllib.parse import unquote

try:
    import bleach
    BLEACH_AVAILABLE = True
except ImportError:
    bleach = None
    BLEACH_AVAILABLE = False
    print("[WARNING] bleach not available for advanced HTML sanitization")

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

class InputSanitizer:
    """
    Comprehensive input sanitization for web applications.
    Handles XSS prevention, SQL injection protection, and input validation.
    """
    
    def __init__(self):
        # XSS patterns (basic protection without bleach)
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'<iframe[^>]*>.*?</iframe>',
            r'<object[^>]*>.*?</object>',
            r'<embed[^>]*>.*?</embed>',
            r'<link[^>]*>',
            r'<meta[^>]*>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'onclick\s*=',
            r'onmouseover\s*=',
            r'onfocus\s*=',
            r'onblur\s*=',
            r'onchange\s*=',
            r'onsubmit\s*=',
        ]
        
        # SQL injection patterns
        self.sql_patterns = [
            r'(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b|\bCREATE\b|\bALTER\b)',
            r'(\b(OR|AND)\s+[\'"]?\d+[\'"]?\s*=\s*[\'"]?\d+[\'"]?)',
            r'(\b(OR|AND)\s+[\'"]?\w+[\'"]?\s*=\s*[\'"]?\w+[\'"]?)',
            r'[\'"][\s]*;[\s]*--',
            r'[\'"][\s]*;[\s]*#',
            r'[\'"][\s]*;[\s]*/\*',
            r'(\bEXEC\b|\bEXECUTE\b)',
            r'(\bxp_\w+)',
            r'(\bsp_\w+)',
        ]
        
        # Command injection patterns
        self.command_patterns = [
            r'[;&|`]',
            r'\$\([^)]*\)',
            r'`[^`]*`',
            r'\|\s*\w+',
            r'>\s*\w+',
            r'<\s*\w+',
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r'\.\./+',
            r'\.\.\\+',
            r'%2e%2e%2f',
            r'%2e%2e\\',
            r'\.\.%2f',
            r'\.\.%5c',
        ]
        
        # Configure bleach if available
        if BLEACH_AVAILABLE:
            self.allowed_tags = ['b', 'i', 'u', 'em', 'strong', 'p', 'br']
            self.allowed_attributes = {}
            self.allowed_protocols = ['http', 'https']
    
    def sanitize_html(self, text: str) -> str:
        """Sanitize HTML content to prevent XSS attacks."""
        if not text:
            return text
        
        if BLEACH_AVAILABLE and bleach:
            # Use bleach for comprehensive sanitization
            return bleach.clean(
                text,
                tags=self.allowed_tags,
                attributes=self.allowed_attributes,
                protocols=self.allowed_protocols,
                strip=True
            )
        else:
            # Fallback to basic sanitization
            # HTML encode all content
            text = html.escape(text)
            
            # Remove dangerous patterns
            for pattern in self.xss_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
            
            return text
    
    def check_sql_injection(self, text: str) -> bool:
        """Check if text contains potential SQL injection patterns."""
        if not text:
            return False
        
        text_upper = text.upper()
        for pattern in self.sql_patterns:
            if re.search(pattern, text_upper, re.IGNORECASE):
                return True
        return False
    
    def check_command_injection(self, text: str) -> bool:
        """Check if text contains potential command injection patterns."""
        if not text:
            return False
        
        for pattern in self.command_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def check_path_traversal(self, text: str) -> bool:
        """Check if text contains path traversal attempts."""
        if not text:
            return False
        
        # URL decode first
        decoded_text = unquote(text)
        
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, decoded_text, re.IGNORECASE):
                return True
        return False
    
    def validate_length(self, text: str, max_length: int = 10000) -> bool:
        """Validate text length to prevent DoS attacks."""
        return len(text) <= max_length
    
    def validate_encoding(self, text: str) -> bool:
        """Validate text encoding to prevent encoding attacks."""
        try:
            # Try to encode and decode to UTF-8
            text.encode('utf-8').decode('utf-8')
            return True
        except UnicodeError:
            return False
    
    def sanitize_user_input(self, text: str, max_length: int = 10000) -> Dict[str, Any]:
        """
        Comprehensive sanitization of user input.
        
        Returns:
            Dictionary with sanitized text and validation results
        """
        if not text:
            return {
                "sanitized_text": "",
                "is_safe": True,
                "warnings": []
            }
        
        warnings = []
        is_safe = True
        
        # Check length
        if not self.validate_length(text, max_length):
            warnings.append(f"Input too long (max {max_length} characters)")
            is_safe = False
            text = text[:max_length]
        
        # Check encoding
        if not self.validate_encoding(text):
            warnings.append("Invalid character encoding detected")
            is_safe = False
            # Try to fix encoding issues
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Check for SQL injection
        if self.check_sql_injection(text):
            warnings.append("Potential SQL injection detected")
            is_safe = False
        
        # Check for command injection
        if self.check_command_injection(text):
            warnings.append("Potential command injection detected")
            is_safe = False
        
        # Check for path traversal
        if self.check_path_traversal(text):
            warnings.append("Potential path traversal detected")
            is_safe = False
        
        # Sanitize HTML
        sanitized_text = self.sanitize_html(text)
        
        # Check if sanitization changed the text significantly
        if len(sanitized_text) < len(text) * 0.8:  # If more than 20% was removed
            warnings.append("Significant content removed during sanitization")
        
        return {
            "sanitized_text": sanitized_text,
            "original_text": text,
            "is_safe": is_safe,
            "warnings": warnings,
            "length": len(sanitized_text)
        }
    
    def sanitize_json_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize JSON data."""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Sanitize the key
                if isinstance(key, str):
                    key_result = self.sanitize_user_input(key, max_length=100)
                    if not key_result["is_safe"]:
                        continue  # Skip unsafe keys
                    sanitized_key = key_result["sanitized_text"]
                else:
                    sanitized_key = key
                
                # Sanitize the value
                sanitized[sanitized_key] = self.sanitize_json_data(value)
            return sanitized
        
        elif isinstance(data, list):
            return [self.sanitize_json_data(item) for item in data]
        
        elif isinstance(data, str):
            result = self.sanitize_user_input(data)
            return result["sanitized_text"]
        
        else:
            return data
    
    def validate_content_type(self, content_type: str) -> bool:
        """Validate request content type."""
        allowed_types = [
            'application/json',
            'multipart/form-data',
            'application/x-www-form-urlencoded',
            'text/plain'
        ]
        
        if content_type:
            main_type = content_type.split(';')[0].strip().lower()
            return main_type in allowed_types
        
        return False
    
    def check_request_headers(self, headers: Dict[str, str]) -> List[str]:
        """Check request headers for potential security issues."""
        warnings = []
        
        # Check for suspicious headers
        suspicious_headers = ['x-forwarded-host', 'x-real-ip', 'x-forwarded-for']
        for header in suspicious_headers:
            if header in headers:
                value = headers[header]
                if self.check_sql_injection(value) or self.check_command_injection(value):
                    warnings.append(f"Suspicious content in header {header}")
        
        # Check User-Agent
        user_agent = headers.get('user-agent', '')
        if user_agent:
            # Check for automated tools/bots that might be malicious
            suspicious_agents = ['sqlmap', 'nmap', 'nikto', 'dirb', 'burp']
            for agent in suspicious_agents:
                if agent.lower() in user_agent.lower():
                    warnings.append(f"Suspicious user agent: {agent}")
        
        return warnings


class SecurityMiddleware:
    """FastAPI middleware for input sanitization and security."""
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.blocked_ips = set()
        self.suspicious_requests = {}  # IP -> count
        self.max_suspicious_requests = 10
    
    async def __call__(self, request: Request, call_next):
        """Process request through security middleware."""
        start_time = time.time()
        
        try:
            # Get client IP
            client_ip = self._get_client_ip(request)
            
            # Check if IP is blocked
            if client_ip in self.blocked_ips:
                return JSONResponse(
                    status_code=403,
                    content={"error": "Access denied", "message": "IP blocked due to suspicious activity"}
                )
            
            # Validate request headers
            header_warnings = self.sanitizer.check_request_headers(dict(request.headers))
            if header_warnings:
                self._record_suspicious_activity(client_ip)
                # Log warnings but don't block for headers
                print(f"Header warnings for {client_ip}: {header_warnings}")
            
            # Check content type for POST requests
            if request.method in ['POST', 'PUT', 'PATCH']:
                content_type = request.headers.get('content-type', '')
                if not self.sanitizer.validate_content_type(content_type):
                    self._record_suspicious_activity(client_ip)
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Invalid content type", "message": "Unsupported content type"}
                    )
            
            # Process the request
            response = await call_next(request)
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            
            # Add response time header for monitoring
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(round(process_time, 4))
            
            return response
            
        except Exception as e:
            print(f"Security middleware error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal security error"}
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded IPs (reverse proxy scenarios)
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # Fallback to direct connection IP
        return request.client.host if request.client else 'unknown'
    
    def _record_suspicious_activity(self, ip: str):
        """Record suspicious activity and block if threshold exceeded."""
        self.suspicious_requests[ip] = self.suspicious_requests.get(ip, 0) + 1
        
        if self.suspicious_requests[ip] >= self.max_suspicious_requests:
            self.blocked_ips.add(ip)
            print(f"🚫 IP {ip} blocked due to {self.suspicious_requests[ip]} suspicious requests")
    
    def unblock_ip(self, ip: str):
        """Manually unblock an IP address."""
        self.blocked_ips.discard(ip)
        self.suspicious_requests.pop(ip, None)
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get current security statistics."""
        return {
            "blocked_ips": list(self.blocked_ips),
            "suspicious_activity": dict(self.suspicious_requests),
            "total_blocked": len(self.blocked_ips),
            "total_suspicious": len(self.suspicious_requests)
        }


# Global instances
input_sanitizer = InputSanitizer()
security_middleware = SecurityMiddleware()

def sanitize_text(text: str, max_length: int = 10000) -> str:
    """Quick function to sanitize text input."""
    result = input_sanitizer.sanitize_user_input(text, max_length)
    return result["sanitized_text"]

def is_safe_input(text: str) -> bool:
    """Quick function to check if input is safe."""
    result = input_sanitizer.sanitize_user_input(text)
    return result["is_safe"]

def sanitize_request_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Quick function to sanitize request data."""
    return input_sanitizer.sanitize_json_data(data)

# Import time for middleware
import time
