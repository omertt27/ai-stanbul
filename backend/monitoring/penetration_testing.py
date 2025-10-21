"""
Penetration Testing Framework for AI Istanbul
Comprehensive security testing with automated vulnerability scanning
"""

import asyncio
import aiohttp
import json
import time
import random
import string
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urljoin, urlparse
import base64
import hashlib
from enum import Enum
import ssl
import socket
from pathlib import Path

class VulnerabilityLevel(Enum):
    """Vulnerability severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TestCategory(Enum):
    """Penetration test categories"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INPUT_VALIDATION = "input_validation"
    INJECTION = "injection"
    XSS = "xss"
    CSRF = "csrf"
    SESSION_MANAGEMENT = "session_management"
    INFORMATION_DISCLOSURE = "information_disclosure"
    RATE_LIMITING = "rate_limiting"
    SSL_TLS = "ssl_tls"
    CONFIGURATION = "configuration"

@dataclass
class VulnerabilityReport:
    """Individual vulnerability report"""
    test_name: str
    category: TestCategory
    severity: VulnerabilityLevel
    description: str
    endpoint: str
    method: str
    payload: Optional[str] = None
    response_status: Optional[int] = None
    response_headers: Optional[Dict[str, str]] = None
    response_body: Optional[str] = None
    evidence: Optional[str] = None
    recommendation: Optional[str] = None
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class PenTestResults:
    """Complete penetration test results"""
    start_time: str
    end_time: str
    target_url: str
    total_tests: int
    vulnerabilities: List[VulnerabilityReport] = field(default_factory=list)
    test_summary: Dict[str, int] = field(default_factory=dict)
    
    def get_severity_counts(self) -> Dict[str, int]:
        """Get count of vulnerabilities by severity"""
        counts = {level.value: 0 for level in VulnerabilityLevel}
        for vuln in self.vulnerabilities:
            counts[vuln.severity.value] += 1
        return counts

class AIIstanbulPenTester:
    """Penetration tester specifically for AI Istanbul application"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None
        self.results = None
        self.auth_token = None
        self.session_cookies = None
        
        # Common payloads for testing
        self.sql_payloads = [
            "' OR '1'='1",
            "' OR '1'='1' --",
            "' OR '1'='1' /*",
            "admin'--",
            "admin'/*",
            "' OR 1=1#",
            "' UNION SELECT 1,2,3--",
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "1 OR 1=1"
        ]
        
        self.xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src=javascript:alert('XSS')></iframe>",
            "'><script>alert('XSS')</script>",
            "\"><script>alert('XSS')</script>",
            "<script>document.write('XSS')</script>",
            "<body onload=alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>"
        ]
        
        self.command_injection_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& whoami",
            "; cat /etc/hosts",
            "| id",
            "; ps aux",
            "&& uname -a",
            "| netstat -an",
            "; env",
            "&& pwd"
        ]
        
        self.path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd"
        ]
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def run_comprehensive_test(self) -> PenTestResults:
        """Run comprehensive penetration test suite"""
        start_time = datetime.utcnow()
        self.results = PenTestResults(
            start_time=start_time.isoformat(),
            end_time="",
            target_url=self.base_url,
            total_tests=0
        )
        
        print(f"🔍 Starting comprehensive penetration test for: {self.base_url}")
        
        # Test categories to run
        test_methods = [
            self.test_ssl_configuration,
            self.test_http_security_headers,
            self.test_information_disclosure,
            self.test_authentication_bypass,
            self.test_authorization_flaws,
            self.test_input_validation,
            self.test_sql_injection,
            self.test_xss_vulnerabilities,
            self.test_command_injection,
            self.test_path_traversal,
            self.test_rate_limiting,
            self.test_session_management,
            self.test_csrf_protection,
            self.test_api_security,
            self.test_admin_interfaces,
            self.test_error_handling,
        ]
        
        for test_method in test_methods:
            try:
                print(f"  Running: {test_method.__name__}")
                await test_method()
                self.results.total_tests += 1
            except Exception as e:
                print(f"  ❌ Test {test_method.__name__} failed: {e}")
        
        end_time = datetime.utcnow()
        self.results.end_time = end_time.isoformat()
        
        # Generate summary
        self.results.test_summary = self.results.get_severity_counts()
        
        print(f"\n✅ Penetration test completed in {(end_time - start_time).total_seconds():.2f} seconds")
        print(f"   Found {len(self.results.vulnerabilities)} potential vulnerabilities")
        
        return self.results
    
    async def test_ssl_configuration(self):
        """Test SSL/TLS configuration"""
        if not self.base_url.startswith('https://'):
            self.add_vulnerability(
                "HTTP_ONLY_ACCESS",
                TestCategory.SSL_TLS,
                VulnerabilityLevel.HIGH,
                "Application accessible over HTTP instead of HTTPS",
                self.base_url,
                "GET",
                recommendation="Implement HTTPS redirects and HSTS headers",
                cwe_id="CWE-319"
            )
            return
        
        # Test SSL certificate
        try:
            hostname = urlparse(self.base_url).hostname
            context = ssl.create_default_context()
            
            with socket.create_connection((hostname, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    
                    # Check certificate validity
                    not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    days_until_expiry = (not_after - datetime.utcnow()).days
                    
                    if days_until_expiry < 30:
                        self.add_vulnerability(
                            "SSL_CERT_EXPIRING",
                            TestCategory.SSL_TLS,
                            VulnerabilityLevel.MEDIUM,
                            f"SSL certificate expires in {days_until_expiry} days",
                            self.base_url,
                            "CERT_CHECK",
                            recommendation="Renew SSL certificate before expiration"
                        )
        
        except Exception as e:
            self.add_vulnerability(
                "SSL_CONFIG_ERROR",
                TestCategory.SSL_TLS,
                VulnerabilityLevel.MEDIUM,
                f"SSL configuration issue: {str(e)}",
                self.base_url,
                "SSL_CHECK",
                recommendation="Review SSL/TLS configuration"
            )
    
    async def test_http_security_headers(self):
        """Test for security headers"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                headers = response.headers
                
                # Required security headers
                security_headers = {
                    'X-Content-Type-Options': 'nosniff',
                    'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
                    'X-XSS-Protection': '1; mode=block',
                    'Strict-Transport-Security': 'max-age=',
                    'Content-Security-Policy': '',
                    'Referrer-Policy': 'strict-origin-when-cross-origin'
                }
                
                for header, expected in security_headers.items():
                    if header not in headers:
                        self.add_vulnerability(
                            f"MISSING_SECURITY_HEADER_{header.upper()}",
                            TestCategory.CONFIGURATION,
                            VulnerabilityLevel.MEDIUM,
                            f"Missing security header: {header}",
                            f"{self.base_url}/health",
                            "GET",
                            recommendation=f"Add {header} header to all responses",
                            cwe_id="CWE-1021"
                        )
                    elif isinstance(expected, list):
                        if not any(exp in str(headers[header]) for exp in expected):
                            self.add_vulnerability(
                                f"WEAK_SECURITY_HEADER_{header.upper()}",
                                TestCategory.CONFIGURATION,
                                VulnerabilityLevel.LOW,
                                f"Weak {header} header configuration",
                                f"{self.base_url}/health",
                                "GET",
                                recommendation=f"Strengthen {header} header configuration"
                            )
                    elif expected and expected not in str(headers.get(header, '')):
                        self.add_vulnerability(
                            f"WEAK_SECURITY_HEADER_{header.upper()}",
                            TestCategory.CONFIGURATION,
                            VulnerabilityLevel.LOW,
                            f"Weak {header} header configuration",
                            f"{self.base_url}/health",
                            "GET",
                            recommendation=f"Review {header} header configuration"
                        )
        
        except Exception as e:
            print(f"Error testing security headers: {e}")
    
    async def test_information_disclosure(self):
        """Test for information disclosure vulnerabilities"""
        # Test for server information disclosure
        endpoints_to_test = [
            "/health",
            "/docs",
            "/redoc", 
            "/openapi.json",
            "/.env",
            "/config",
            "/admin",
            "/api/stats",
            "/server-status",
            "/info.php",
            "/phpinfo.php",
            "/.git/config",
            "/robots.txt",
            "/sitemap.xml"
        ]
        
        for endpoint in endpoints_to_test:
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    if response.status == 200:
                        body = await response.text()
                        
                        # Check for sensitive information patterns
                        sensitive_patterns = [
                            ("password", "Potential password disclosure"),
                            ("secret", "Potential secret disclosure"),
                            ("api_key", "API key disclosure"),
                            ("token", "Token disclosure"),
                            ("database", "Database information disclosure"),
                            ("internal", "Internal system information"),
                            ("debug", "Debug information disclosure"),
                            ("error", "Error information disclosure")
                        ]
                        
                        for pattern, description in sensitive_patterns:
                            if pattern.lower() in body.lower():
                                self.add_vulnerability(
                                    f"INFO_DISCLOSURE_{endpoint.replace('/', '_').upper()}",
                                    TestCategory.INFORMATION_DISCLOSURE,
                                    VulnerabilityLevel.LOW,
                                    f"{description} in {endpoint}",
                                    f"{self.base_url}{endpoint}",
                                    "GET",
                                    evidence=f"Found '{pattern}' in response",
                                    recommendation=f"Review {endpoint} for sensitive information exposure",
                                    cwe_id="CWE-200"
                                )
                                break
            
            except Exception:
                continue  # Expected for many endpoints
    
    async def test_authentication_bypass(self):
        """Test for authentication bypass vulnerabilities"""
        # Test admin endpoints without authentication
        admin_endpoints = [
            "/admin/api/stats",
            "/admin/api/sessions", 
            "/admin/api/users",
            "/admin/api/blog/posts",
            "/admin/api/blog/comments"
        ]
        
        for endpoint in admin_endpoints:
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    if response.status != 401 and response.status != 403:
                        self.add_vulnerability(
                            f"AUTH_BYPASS_{endpoint.replace('/', '_').upper()}",
                            TestCategory.AUTHENTICATION,
                            VulnerabilityLevel.HIGH,
                            f"Admin endpoint {endpoint} accessible without authentication",
                            f"{self.base_url}{endpoint}",
                            "GET",
                            response_status=response.status,
                            recommendation="Implement proper authentication for admin endpoints",
                            cwe_id="CWE-306",
                            cvss_score=7.5
                        )
            except Exception:
                continue
    
    async def test_authorization_flaws(self):
        """Test for authorization flaws"""
        # Test for horizontal privilege escalation
        # This would require valid user accounts to test properly
        # For now, test basic authorization patterns
        
        # Test with malformed tokens
        malformed_tokens = [
            "Bearer invalid_token",
            "Bearer ",
            "Bearer null",
            "Bearer undefined",
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.INVALID.SIGNATURE"
        ]
        
        for token in malformed_tokens:
            try:
                headers = {"Authorization": token}
                async with self.session.get(f"{self.base_url}/admin/api/stats", headers=headers) as response:
                    if response.status == 200:
                        self.add_vulnerability(
                            "WEAK_TOKEN_VALIDATION",
                            TestCategory.AUTHORIZATION,
                            VulnerabilityLevel.HIGH,
                            f"Malformed token accepted: {token[:20]}...",
                            f"{self.base_url}/admin/api/stats",
                            "GET",
                            payload=token,
                            recommendation="Implement proper JWT token validation",
                            cwe_id="CWE-287"
                        )
                        break
            except Exception:
                continue
    
    async def test_input_validation(self):
        """Test input validation on various endpoints"""
        # Test chat endpoint with malicious inputs
        test_inputs = [
            {"message": "A" * 10000},  # Very long input
            {"message": "\x00\x01\x02"},  # Null bytes
            {"message": "<?xml version='1.0'?><!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><foo>&xxe;</foo>"},  # XXE
            {"message": json.dumps({"nested": {"deep": {"very": {"deeply": "nested"}}}})},  # Deep nesting
            {"message": "<script>alert('xss')</script>"},  # XSS attempt
        ]
        
        for test_input in test_inputs:
            try:
                async with self.session.post(f"{self.base_url}/ai/chat", json=test_input) as response:
                    if response.status == 500:
                        body = await response.text()
                        self.add_vulnerability(
                            "INPUT_VALIDATION_ERROR",
                            TestCategory.INPUT_VALIDATION,
                            VulnerabilityLevel.MEDIUM,
                            "Input validation error causing server error",
                            f"{self.base_url}/ai/chat",
                            "POST",
                            payload=str(test_input)[:100],
                            response_status=response.status,
                            recommendation="Implement proper input validation and sanitization",
                            cwe_id="CWE-20"
                        )
            except Exception:
                continue
    
    async def test_sql_injection(self):
        """Test for SQL injection vulnerabilities"""
        # Test parameters that might be used in database queries
        test_endpoints = [
            "/restaurants/search?query={payload}&location=Istanbul",
            "/places/search?query={payload}&location=Istanbul",
            "/admin/api/blog/posts?status={payload}",
            "/admin/api/blog/comments?status={payload}"
        ]
        
        for endpoint_template in test_endpoints:
            for payload in self.sql_payloads[:5]:  # Test subset for speed
                endpoint = endpoint_template.format(payload=payload)
                try:
                    async with self.session.get(f"{self.base_url}{endpoint}") as response:
                        body = await response.text()
                        
                        # Look for SQL error patterns
                        sql_error_patterns = [
                            "sql syntax",
                            "mysql error",
                            "postgresql error",
                            "sqlite error",
                            "database error",
                            "sqlstate",
                            "constraint violation"
                        ]
                        
                        for pattern in sql_error_patterns:
                            if pattern.lower() in body.lower():
                                self.add_vulnerability(
                                    "SQL_INJECTION_ERROR",
                                    TestCategory.INJECTION,
                                    VulnerabilityLevel.HIGH,
                                    f"Potential SQL injection vulnerability",
                                    f"{self.base_url}{endpoint}",
                                    "GET",
                                    payload=payload,
                                    evidence=f"SQL error pattern found: {pattern}",
                                    recommendation="Use parameterized queries to prevent SQL injection",
                                    cwe_id="CWE-89",
                                    cvss_score=8.2
                                )
                                break
                except Exception:
                    continue
    
    async def test_xss_vulnerabilities(self):
        """Test for XSS vulnerabilities"""
        # Test various endpoints for XSS
        xss_test_endpoints = [
            {"url": "/ai/chat", "method": "POST", "data": {"message": "{payload}"}},
            {"url": "/restaurants/search", "method": "GET", "params": {"query": "{payload}"}},
            {"url": "/places/search", "method": "GET", "params": {"query": "{payload}"}},
        ]
        
        for endpoint_config in xss_test_endpoints:
            for payload in self.xss_payloads[:3]:  # Test subset for speed
                try:
                    if endpoint_config["method"] == "GET":
                        params = {k: v.format(payload=payload) for k, v in endpoint_config.get("params", {}).items()}
                        async with self.session.get(f"{self.base_url}{endpoint_config['url']}", params=params) as response:
                            body = await response.text()
                    else:
                        data = {k: v.format(payload=payload) for k, v in endpoint_config.get("data", {}).items()}
                        async with self.session.post(f"{self.base_url}{endpoint_config['url']}", json=data) as response:
                            body = await response.text()
                    
                    # Check if payload is reflected unescaped
                    if payload in body and "<script>" in payload:
                        self.add_vulnerability(
                            "XSS_REFLECTED",
                            TestCategory.XSS,
                            VulnerabilityLevel.HIGH,
                            "Potential reflected XSS vulnerability",
                            f"{self.base_url}{endpoint_config['url']}",
                            endpoint_config["method"],
                            payload=payload,
                            evidence="XSS payload reflected in response",
                            recommendation="Implement proper output encoding and CSP headers",
                            cwe_id="CWE-79",
                            cvss_score=7.1
                        )
                        
                except Exception:
                    continue
    
    async def test_command_injection(self):
        """Test for command injection vulnerabilities"""
        # Test endpoints that might execute system commands
        for payload in self.command_injection_payloads[:3]:
            test_data = {"message": f"show me info about Istanbul{payload}"}
            try:
                async with self.session.post(f"{self.base_url}/ai/chat", json=test_data) as response:
                    body = await response.text()
                    
                    # Look for command output patterns
                    command_patterns = [
                        "root:",
                        "/bin/bash",
                        "uid=",
                        "gid=",
                        "kernel version",
                        "total used free"
                    ]
                    
                    for pattern in command_patterns:
                        if pattern.lower() in body.lower():
                            self.add_vulnerability(
                                "COMMAND_INJECTION",
                                TestCategory.INJECTION,
                                VulnerabilityLevel.CRITICAL,
                                "Potential command injection vulnerability",
                                f"{self.base_url}/ai/chat",
                                "POST",
                                payload=payload,
                                evidence=f"Command output pattern found: {pattern}",
                                recommendation="Never execute user input as system commands",
                                cwe_id="CWE-78",
                                cvss_score=9.8
                            )
                            return
            except Exception:
                continue
    
    async def test_path_traversal(self):
        """Test for path traversal vulnerabilities"""
        # Test file access endpoints
        for payload in self.path_traversal_payloads:
            try:
                # Test as query parameter
                async with self.session.get(f"{self.base_url}/admin?file={payload}") as response:
                    body = await response.text()
                    
                    if "root:x:" in body or "# Windows hosts file" in body:
                        self.add_vulnerability(
                            "PATH_TRAVERSAL",
                            TestCategory.INJECTION,
                            VulnerabilityLevel.HIGH,
                            "Path traversal vulnerability detected",
                            f"{self.base_url}/admin",
                            "GET",
                            payload=payload,
                            evidence="System file contents in response",
                            recommendation="Validate file paths and restrict file access",
                            cwe_id="CWE-22",
                            cvss_score=7.5
                        )
                        break
            except Exception:
                continue
    
    async def test_rate_limiting(self):
        """Test rate limiting implementation"""
        # Test chat endpoint rate limiting
        requests_sent = 0
        rate_limited = False
        
        for i in range(60):  # Send 60 requests quickly
            try:
                start_time = time.time()
                async with self.session.post(f"{self.base_url}/ai/chat", 
                                           json={"message": f"test message {i}"}) as response:
                    requests_sent += 1
                    if response.status == 429:  # Too Many Requests
                        rate_limited = True
                        break
                    
                    # If no delay between requests and no rate limiting after many requests
                    if i > 30 and time.time() - start_time < 0.1:
                        continue
                        
            except Exception:
                break
        
        if not rate_limited and requests_sent > 50:
            self.add_vulnerability(
                "MISSING_RATE_LIMITING",
                TestCategory.RATE_LIMITING,
                VulnerabilityLevel.MEDIUM,
                f"No rate limiting detected after {requests_sent} requests",
                f"{self.base_url}/ai/chat",
                "POST",
                recommendation="Implement rate limiting to prevent abuse",
                cwe_id="CWE-770"
            )
    
    async def test_session_management(self):
        """Test session management security"""
        # Test for secure session cookies
        try:
            async with self.session.get(f"{self.base_url}/admin") as response:
                cookies = response.cookies
                
                for cookie_name, cookie in cookies.items():
                    # Check for secure flags
                    if not cookie.get('secure'):
                        self.add_vulnerability(
                            f"INSECURE_COOKIE_{cookie_name.upper()}",
                            TestCategory.SESSION_MANAGEMENT,
                            VulnerabilityLevel.MEDIUM,
                            f"Cookie {cookie_name} missing Secure flag",
                            f"{self.base_url}/admin",
                            "GET",
                            recommendation="Set Secure flag on all cookies",
                            cwe_id="CWE-614"
                        )
                    
                    if not cookie.get('httponly'):
                        self.add_vulnerability(
                            f"HTTPONLY_COOKIE_{cookie_name.upper()}",
                            TestCategory.SESSION_MANAGEMENT,
                            VulnerabilityLevel.LOW,
                            f"Cookie {cookie_name} missing HttpOnly flag",
                            f"{self.base_url}/admin",
                            "GET",
                            recommendation="Set HttpOnly flag on session cookies",
                            cwe_id="CWE-1004"
                        )
        except Exception:
            pass
    
    async def test_csrf_protection(self):
        """Test CSRF protection"""
        # Test state-changing operations without CSRF tokens
        csrf_test_endpoints = [
            {"url": "/auth/login", "method": "POST", "data": {"username": "test", "password": "test"}},
            {"url": "/admin/api/blog/posts/1/moderate", "method": "POST", "data": {"action": "publish"}},
        ]
        
        for endpoint_config in csrf_test_endpoints:
            try:
                if endpoint_config["method"] == "POST":
                    # Test without referrer header (potential CSRF)
                    headers = {"Origin": "https://malicious-site.com"}
                    async with self.session.post(
                        f"{self.base_url}{endpoint_config['url']}", 
                        json=endpoint_config["data"],
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            self.add_vulnerability(
                                "CSRF_PROTECTION_MISSING",
                                TestCategory.CSRF,
                                VulnerabilityLevel.MEDIUM,
                                f"CSRF protection may be missing on {endpoint_config['url']}",
                                f"{self.base_url}{endpoint_config['url']}",
                                "POST",
                                recommendation="Implement CSRF tokens for state-changing operations",
                                cwe_id="CWE-352"
                            )
            except Exception:
                continue
    
    async def test_api_security(self):
        """Test API-specific security issues"""
        # Test for verbose error responses
        try:
            async with self.session.post(f"{self.base_url}/ai/chat", 
                                       json={"invalid": "data"}) as response:
                body = await response.text()
                
                # Check for verbose error information
                if "traceback" in body.lower() or "exception" in body.lower():
                    self.add_vulnerability(
                        "VERBOSE_ERROR_MESSAGES",
                        TestCategory.INFORMATION_DISCLOSURE,
                        VulnerabilityLevel.LOW,
                        "Verbose error messages may leak sensitive information",
                        f"{self.base_url}/ai/chat",
                        "POST",
                        recommendation="Implement generic error messages for production",
                        cwe_id="CWE-209"
                    )
        except Exception:
            pass
    
    async def test_admin_interfaces(self):
        """Test admin interface security"""
        # Test for default credentials
        default_creds = [
            ("admin", "admin"),
            ("admin", "password"),
            ("admin", "123456"),
            ("administrator", "administrator"),
            ("root", "root")
        ]
        
        for username, password in default_creds:
            try:
                login_data = {"username": username, "password": password}
                async with self.session.post(f"{self.base_url}/auth/login", json=login_data) as response:
                    if response.status == 200:
                        self.add_vulnerability(
                            f"DEFAULT_CREDENTIALS_{username.upper()}",
                            TestCategory.AUTHENTICATION,
                            VulnerabilityLevel.CRITICAL,
                            f"Default credentials accepted: {username}/{password}",
                            f"{self.base_url}/auth/login",
                            "POST",
                            payload=f"{username}:{password}",
                            recommendation="Change all default credentials immediately",
                            cwe_id="CWE-521",
                            cvss_score=9.8
                        )
            except Exception:
                continue
    
    async def test_error_handling(self):
        """Test error handling for information leakage"""
        # Send malformed requests to trigger errors
        malformed_requests = [
            {"url": "/ai/chat", "method": "POST", "data": "invalid json"},
            {"url": "/ai/chat", "method": "POST", "headers": {"Content-Type": "application/xml"}},
            {"url": "/nonexistent/endpoint", "method": "GET"}
        ]
        
        for request_config in malformed_requests:
            try:
                if request_config["method"] == "POST":
                    headers = request_config.get("headers", {})
                    async with self.session.post(
                        f"{self.base_url}{request_config['url']}", 
                        data=request_config.get("data"),
                        headers=headers
                    ) as response:
                        body = await response.text()
                        
                        # Check for information leakage in error responses
                        sensitive_info = ["file path", "database", "internal", "stack trace"]
                        for info in sensitive_info:
                            if info in body.lower():
                                self.add_vulnerability(
                                    "ERROR_INFO_LEAKAGE",
                                    TestCategory.INFORMATION_DISCLOSURE,
                                    VulnerabilityLevel.LOW,
                                    f"Error response contains sensitive information: {info}",
                                    f"{self.base_url}{request_config['url']}",
                                    request_config["method"],
                                    recommendation="Implement generic error messages",
                                    cwe_id="CWE-209"
                                )
                                break
            except Exception:
                continue
    
    def add_vulnerability(self, test_name: str, category: TestCategory, severity: VulnerabilityLevel,
                         description: str, endpoint: str, method: str, **kwargs):
        """Add a vulnerability to the results"""
        vulnerability = VulnerabilityReport(
            test_name=test_name,
            category=category,
            severity=severity,
            description=description,
            endpoint=endpoint,
            method=method,
            **kwargs
        )
        self.results.vulnerabilities.append(vulnerability)
    
    def generate_report(self, output_file: str = "pentest_report.json"):
        """Generate comprehensive penetration test report"""
        if not self.results:
            return None
        
        report_data = {
            "test_metadata": {
                "target_url": self.results.target_url,
                "start_time": self.results.start_time,
                "end_time": self.results.end_time,
                "total_tests": self.results.total_tests,
                "total_vulnerabilities": len(self.results.vulnerabilities)
            },
            "severity_summary": self.results.get_severity_counts(),
            "vulnerabilities": [
                {
                    "id": i+1,
                    "test_name": vuln.test_name,
                    "category": vuln.category.value,
                    "severity": vuln.severity.value,
                    "description": vuln.description,
                    "endpoint": vuln.endpoint,
                    "method": vuln.method,
                    "payload": vuln.payload,
                    "evidence": vuln.evidence,
                    "recommendation": vuln.recommendation,
                    "cwe_id": vuln.cwe_id,
                    "cvss_score": vuln.cvss_score,
                    "timestamp": vuln.timestamp
                }
                for i, vuln in enumerate(self.results.vulnerabilities)
            ]
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return report_data

async def run_penetration_test(target_url: str, output_file: str = "pentest_report.json"):
    """Run penetration test and generate report"""
    async with AIIstanbulPenTester(target_url) as pen_tester:
        results = await pen_tester.run_comprehensive_test()
        report = pen_tester.generate_report(output_file)
        return report

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python penetration_testing.py <target_url> [output_file]")
        sys.exit(1)
    
    target_url = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "pentest_report.json"
    
    # Run penetration test
    report = asyncio.run(run_penetration_test(target_url, output_file))
    
    if report:
        print(f"\n📊 Penetration Test Report Summary")
        print(f"Target: {report['test_metadata']['target_url']}")
        print(f"Total Tests: {report['test_metadata']['total_tests']}")
        print(f"Total Vulnerabilities: {report['test_metadata']['total_vulnerabilities']}")
        print(f"\nSeverity Breakdown:")
        for severity, count in report['severity_summary'].items():
            if count > 0:
                print(f"  {severity.upper()}: {count}")
        print(f"\nReport saved to: {output_file}")
