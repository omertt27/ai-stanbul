"""
Enhanced Security Module for AI Istanbul Backend
Provides authentication, rate limiting, input validation, and security monitoring
"""

import hashlib
import hmac
import time
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from functools import wraps
import redis
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from passlib.context import CryptContext

# Security Configuration
class SecurityConfig:
    SECRET_KEY = "your-secret-key-change-in-production"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS = 100  # requests per window
    RATE_LIMIT_WINDOW = 3600   # 1 hour in seconds
    RATE_LIMIT_BURST = 20      # burst limit
    
    # Security Thresholds
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION = 900     # 15 minutes
    MAX_REQUEST_SIZE = 10000   # bytes
    SUSPICIOUS_PATTERN_THRESHOLD = 10

@dataclass
class SecurityEvent:
    timestamp: str
    event_type: str
    severity: str  # low, medium, high, critical
    source_ip: str
    user_agent: str
    details: Dict
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class SecurityMonitor:
    """Real-time security monitoring and threat detection"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.suspicious_patterns = [
            r'union\s+select',
            r'drop\s+table', 
            r'<script.*?>',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'\.\./.*\.\./',  # Path traversal
            r'cmd\.exe',
            r'/etc/passwd'
        ]
        self.blocked_ips: Set[str] = set()
        self.rate_limit_cache: Dict[str, List[float]] = {}
        
    def log_security_event(self, event: SecurityEvent):
        """Log security events for analysis"""
        try:
            if self.redis_client:
                key = f"security:events:{datetime.now().strftime('%Y%m%d')}"
                event_data = json.dumps(asdict(event))
                self.redis_client.lpush(key, event_data)
                self.redis_client.expire(key, 86400 * 7)  # Keep for 7 days
                
                # Count events by type for alerting
                counter_key = f"security:counter:{event.event_type}:{datetime.now().strftime('%Y%m%d%H')}"
                count = self.redis_client.incr(counter_key)
                self.redis_client.expire(counter_key, 3600)
                
                # Alert on high-severity events or unusual patterns
                if event.severity in ['high', 'critical'] or count > 50:
                    self._trigger_security_alert(event, count)
                    
        except Exception as e:
            print(f"⚠️ Security logging error: {e}")
    
    def _trigger_security_alert(self, event: SecurityEvent, count: int):
        """Trigger security alerts for serious events"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'event': asdict(event),
            'frequency': count,
            'alert_level': 'HIGH' if event.severity == 'critical' else 'MEDIUM'
        }
        
        # In production, this would send to monitoring systems
        print(f"🚨 SECURITY ALERT: {event.event_type} - {event.severity.upper()}")
        print(f"   Source: {event.source_ip}")
        print(f"   Count: {count} in last hour")
    
    def is_suspicious_request(self, content: str, request: Request) -> bool:
        """Detect suspicious request patterns"""
        if not content:
            return False
            
        # Check for malicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                self.log_security_event(SecurityEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="suspicious_pattern",
                    severity="high",
                    source_ip=request.client.host if request.client else "unknown",
                    user_agent=request.headers.get("user-agent", "unknown"),
                    details={"pattern": pattern, "content_preview": content[:100]}
                ))
                return True
        
        # Check for unusual request patterns
        if len(content) > SecurityConfig.MAX_REQUEST_SIZE:
            self.log_security_event(SecurityEvent(
                timestamp=datetime.now().isoformat(),
                event_type="oversized_request",
                severity="medium",
                source_ip=request.client.host if request.client else "unknown",
                user_agent=request.headers.get("user-agent", "unknown"),
                details={"size": len(content)}
            ))
            return True
            
        return False
    
    def check_rate_limit(self, identifier: str, request: Request) -> bool:
        """Advanced rate limiting with burst protection"""
        now = time.time()
        window_start = now - SecurityConfig.RATE_LIMIT_WINDOW
        
        # Clean old entries
        if identifier in self.rate_limit_cache:
            self.rate_limit_cache[identifier] = [
                timestamp for timestamp in self.rate_limit_cache[identifier]
                if timestamp > window_start
            ]
        else:
            self.rate_limit_cache[identifier] = []
        
        request_times = self.rate_limit_cache[identifier]
        
        # Check burst limit (requests in last minute)
        recent_requests = [t for t in request_times if t > (now - 60)]
        if len(recent_requests) > SecurityConfig.RATE_LIMIT_BURST:
            self.log_security_event(SecurityEvent(
                timestamp=datetime.now().isoformat(),
                event_type="rate_limit_burst",
                severity="medium",
                source_ip=request.client.host if request.client else "unknown",
                user_agent=request.headers.get("user-agent", "unknown"),
                details={"requests_per_minute": len(recent_requests)}
            ))
            return False
        
        # Check overall rate limit
        if len(request_times) >= SecurityConfig.RATE_LIMIT_REQUESTS:
            self.log_security_event(SecurityEvent(
                timestamp=datetime.now().isoformat(),
                event_type="rate_limit_exceeded",
                severity="medium",
                source_ip=request.client.host if request.client else "unknown",
                user_agent=request.headers.get("user-agent", "unknown"),
                details={"requests_per_hour": len(request_times)}
            ))
            return False
        
        # Add current request
        request_times.append(now)
        self.rate_limit_cache[identifier] = request_times
        
        return True
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        return ip in self.blocked_ips
    
    def block_ip(self, ip: str, duration: int = 3600):
        """Block IP for specified duration"""
        self.blocked_ips.add(ip)
        
        # Remove after duration (in production, use Redis with TTL)
        if self.redis_client:
            self.redis_client.setex(f"blocked_ip:{ip}", duration, "1")

class AuthenticationManager:
    """JWT-based authentication with enhanced security"""
    
    def __init__(self, redis_client=None):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.redis_client = redis_client
        self.failed_attempts: Dict[str, List[float]] = {}
        
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password with timing attack protection"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash password securely"""
        return self.pwd_context.hash(password)
    
    def create_access_token(self, data: dict) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire, "type": "access"})
        
        return jwt.encode(to_encode, SecurityConfig.SECRET_KEY, algorithm=SecurityConfig.ALGORITHM)
    
    def create_refresh_token(self, data: dict) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=SecurityConfig.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        return jwt.encode(to_encode, SecurityConfig.SECRET_KEY, algorithm=SecurityConfig.ALGORITHM)
    
    def verify_token(self, token: str) -> Dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SecurityConfig.SECRET_KEY, algorithms=[SecurityConfig.ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def check_login_attempts(self, identifier: str, request: Request) -> bool:
        """Check failed login attempts with lockout"""
        now = time.time()
        window_start = now - SecurityConfig.LOCKOUT_DURATION
        
        if identifier in self.failed_attempts:
            self.failed_attempts[identifier] = [
                timestamp for timestamp in self.failed_attempts[identifier]
                if timestamp > window_start
            ]
        else:
            self.failed_attempts[identifier] = []
        
        failed_count = len(self.failed_attempts[identifier])
        
        if failed_count >= SecurityConfig.MAX_LOGIN_ATTEMPTS:
            # Log security event
            security_monitor.log_security_event(SecurityEvent(
                timestamp=datetime.now().isoformat(),
                event_type="login_lockout",
                severity="high",
                source_ip=request.client.host if request.client else "unknown",
                user_agent=request.headers.get("user-agent", "unknown"),
                details={"failed_attempts": failed_count, "identifier": identifier}
            ))
            return False
        
        return True
    
    def record_failed_login(self, identifier: str):
        """Record failed login attempt"""
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        self.failed_attempts[identifier].append(time.time())

class InputValidator:
    """Advanced input validation and sanitization"""
    
    @staticmethod
    def validate_chat_message(message: str) -> str:
        """Validate and sanitize chat messages"""
        if not message or not isinstance(message, str):
            raise HTTPException(status_code=400, detail="Invalid message format")
        
        # Length validation
        if len(message) > SecurityConfig.MAX_REQUEST_SIZE:
            raise HTTPException(status_code=400, detail="Message too long")
        
        # Remove potentially dangerous content
        message = re.sub(r'<script.*?</script>', '', message, flags=re.IGNORECASE | re.DOTALL)
        message = re.sub(r'javascript:', '', message, flags=re.IGNORECASE)
        message = re.sub(r'vbscript:', '', message, flags=re.IGNORECASE)
        
        # Normalize whitespace
        message = ' '.join(message.split())
        
        return message.strip()
    
    @staticmethod
    def validate_coordinates(lat: float, lng: float) -> bool:
        """Validate GPS coordinates"""
        # Istanbul bounds (approximate)
        ISTANBUL_BOUNDS = {
            'lat_min': 40.8,
            'lat_max': 41.3,
            'lng_min': 28.5,
            'lng_max': 29.5
        }
        
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            return False
        
        # For Istanbul-specific app, coordinates should be within reasonable bounds
        if not (ISTANBUL_BOUNDS['lat_min'] <= lat <= ISTANBUL_BOUNDS['lat_max'] and
                ISTANBUL_BOUNDS['lng_min'] <= lng <= ISTANBUL_BOUNDS['lng_max']):
            # Log suspicious coordinates but don't reject (could be legitimate)
            print(f"⚠️ Coordinates outside Istanbul bounds: {lat}, {lng}")
        
        return True
    
    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        """Validate session ID format"""
        if not session_id or len(session_id) > 100:
            return False
        
        # Allow alphanumeric, hyphens, underscores
        return re.match(r'^[a-zA-Z0-9_-]+$', session_id) is not None

# Initialize security components
security_monitor = SecurityMonitor()
auth_manager = AuthenticationManager()
security_bearer = HTTPBearer(auto_error=False)

# Security middleware and decorators
def security_middleware(request: Request):
    """Security middleware for all requests"""
    client_ip = request.client.host if request.client else "unknown"
    
    # Check if IP is blocked
    if security_monitor.is_ip_blocked(client_ip):
        raise HTTPException(status_code=403, detail="IP address blocked")
    
    # Rate limiting
    rate_limit_key = f"{client_ip}:{request.url.path}"
    if not security_monitor.check_rate_limit(rate_limit_key, request):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    return True

def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security_bearer)):
    """Require valid JWT authentication"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        payload = auth_manager.verify_token(credentials.credentials)
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication")

def validate_request_content(request_data: dict, request: Request) -> dict:
    """Validate and sanitize request content"""
    # Check for suspicious patterns
    content_str = json.dumps(request_data)
    if security_monitor.is_suspicious_request(content_str, request):
        raise HTTPException(status_code=400, detail="Request contains suspicious content")
    
    # Validate specific fields
    if 'message' in request_data:
        request_data['message'] = InputValidator.validate_chat_message(request_data['message'])
    
    if 'session_id' in request_data and request_data['session_id']:
        if not InputValidator.validate_session_id(request_data['session_id']):
            raise HTTPException(status_code=400, detail="Invalid session ID format")
    
    # Validate location coordinates if present
    location_context = request_data.get('location_context')
    if location_context and location_context.get('has_location'):
        lat = location_context.get('latitude')
        lng = location_context.get('longitude')
        if lat is not None and lng is not None:
            if not InputValidator.validate_coordinates(float(lat), float(lng)):
                raise HTTPException(status_code=400, detail="Invalid coordinates")
    
    return request_data

# Security reporting functions
def get_security_stats() -> Dict:
    """Get security statistics"""
    if security_monitor.redis_client:
        try:
            today = datetime.now().strftime('%Y%m%d')
            events_key = f"security:events:{today}"
            
            event_count = security_monitor.redis_client.llen(events_key)
            blocked_ips = len(security_monitor.blocked_ips)
            
            # Get event types
            events = security_monitor.redis_client.lrange(events_key, 0, -1)
            event_types = {}
            
            for event_data in events:
                try:
                    event = json.loads(event_data)
                    event_type = event.get('event_type', 'unknown')
                    event_types[event_type] = event_types.get(event_type, 0) + 1
                except:
                    continue
            
            return {
                'total_events_today': event_count,
                'blocked_ips': blocked_ips,
                'event_types': event_types,
                'rate_limit_active_windows': len(security_monitor.rate_limit_cache),
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    return {
        'total_events_today': 0,
        'blocked_ips': len(security_monitor.blocked_ips),
        'event_types': {},
        'rate_limit_active_windows': len(security_monitor.rate_limit_cache),
        'redis_available': False
    }
