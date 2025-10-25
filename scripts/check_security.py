#!/usr/bin/env python3
"""
Security Configuration Verification Script
Checks and validates security settings for production deployment
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class SecurityChecker:
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.checks_warning = 0
        self.results = []
        
    def check(self, name: str, condition: bool, message: str, severity: str = "error"):
        """Record a check result"""
        if condition:
            self.checks_passed += 1
            status = f"{GREEN}✓ PASS{RESET}"
            self.results.append({"name": name, "status": "pass", "message": message})
        else:
            if severity == "warning":
                self.checks_warning += 1
                status = f"{YELLOW}⚠ WARN{RESET}"
            else:
                self.checks_failed += 1
                status = f"{RED}✗ FAIL{RESET}"
            self.results.append({"name": name, "status": severity, "message": message})
        
        print(f"{status} {name}: {message}")
        
    def section(self, title: str):
        """Print a section header"""
        print(f"\n{BLUE}{BOLD}{'='*60}{RESET}")
        print(f"{BLUE}{BOLD}{title}{RESET}")
        print(f"{BLUE}{BOLD}{'='*60}{RESET}\n")
    
    def summary(self):
        """Print summary of checks"""
        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}SECURITY CHECK SUMMARY{RESET}")
        print(f"{BOLD}{'='*60}{RESET}")
        print(f"{GREEN}Passed: {self.checks_passed}{RESET}")
        print(f"{YELLOW}Warnings: {self.checks_warning}{RESET}")
        print(f"{RED}Failed: {self.checks_failed}{RESET}")
        print(f"{BOLD}{'='*60}{RESET}\n")
        
        if self.checks_failed > 0:
            print(f"{RED}{BOLD}⚠️  CRITICAL: {self.checks_failed} security checks failed!{RESET}")
            print(f"{RED}Please fix these issues before deploying to production.{RESET}\n")
            return False
        elif self.checks_warning > 0:
            print(f"{YELLOW}{BOLD}⚠️  WARNING: {self.checks_warning} checks need attention.{RESET}")
            print(f"{YELLOW}Review these warnings before production deployment.{RESET}\n")
            return True
        else:
            print(f"{GREEN}{BOLD}✅ All security checks passed!{RESET}\n")
            return True


def check_environment_variables(checker: SecurityChecker):
    """Check environment variables configuration"""
    checker.section("1. ENVIRONMENT VARIABLES")
    
    # Check if .env file exists
    env_file = Path("backend/.env")
    checker.check(
        "ENV File Exists",
        env_file.exists(),
        ".env file found in backend/" if env_file.exists() else ".env file not found! Create from .env.example",
        "error"
    )
    
    if not env_file.exists():
        return
    
    # Load .env file
    env_vars = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
    
    # Check JWT_SECRET_KEY
    jwt_secret = env_vars.get('JWT_SECRET_KEY', '')
    checker.check(
        "JWT_SECRET_KEY Set",
        bool(jwt_secret and jwt_secret != 'your_jwt_secret_key_here_change_this'),
        f"JWT secret key is configured" if jwt_secret and jwt_secret != 'your_jwt_secret_key_here_change_this' 
        else "JWT_SECRET_KEY needs to be changed from default!",
        "error"
    )
    
    checker.check(
        "JWT_SECRET_KEY Length",
        len(jwt_secret) >= 32,
        f"JWT secret key length: {len(jwt_secret)} characters" if len(jwt_secret) >= 32
        else f"JWT secret key too short ({len(jwt_secret)} chars). Should be >= 32 characters",
        "warning"
    )
    
    # Check SECRET_KEY
    secret_key = env_vars.get('SECRET_KEY', '')
    checker.check(
        "SECRET_KEY Set",
        bool(secret_key and secret_key != 'your_super_secret_key_for_production_here_change_this'),
        f"Secret key is configured" if secret_key and secret_key != 'your_super_secret_key_for_production_here_change_this'
        else "SECRET_KEY needs to be changed from default!",
        "error"
    )
    
    # Check DATABASE_URL
    db_url = env_vars.get('DATABASE_URL', '')
    checker.check(
        "DATABASE_URL Set",
        bool(db_url and 'localhost' not in db_url),
        "Database URL configured for production" if db_url and 'localhost' not in db_url
        else "DATABASE_URL should point to production database (not localhost)",
        "warning"
    )
    
    # Check ENVIRONMENT
    environment = env_vars.get('ENVIRONMENT', '')
    checker.check(
        "ENVIRONMENT Set",
        environment == 'production',
        f"Environment: {environment}" if environment == 'production'
        else f"ENVIRONMENT should be 'production' (current: {environment})",
        "warning"
    )
    
    # Check DEBUG mode
    debug = env_vars.get('DEBUG', 'False')
    checker.check(
        "DEBUG Mode",
        debug.lower() in ['false', '0', 'no'],
        "DEBUG is disabled" if debug.lower() in ['false', '0', 'no']
        else "DEBUG should be False in production!",
        "error"
    )
    
    # Check RATE_LIMIT_ENABLED
    rate_limit = env_vars.get('RATE_LIMIT_ENABLED', 'False')
    checker.check(
        "Rate Limiting",
        rate_limit.lower() in ['true', '1', 'yes'],
        "Rate limiting is enabled" if rate_limit.lower() in ['true', '1', 'yes']
        else "Rate limiting should be enabled for production",
        "error"
    )


def check_file_permissions(checker: SecurityChecker):
    """Check file permissions and security"""
    checker.section("2. FILE SECURITY")
    
    # Check if .env is in .gitignore
    gitignore_file = Path(".gitignore")
    if gitignore_file.exists():
        with open(gitignore_file) as f:
            gitignore_content = f.read()
        checker.check(
            ".env in .gitignore",
            '.env' in gitignore_content,
            ".env file is excluded from git" if '.env' in gitignore_content
            else ".env should be added to .gitignore!",
            "error"
        )
    else:
        checker.check(
            ".gitignore exists",
            False,
            ".gitignore file not found!",
            "warning"
        )
    
    # Check for hardcoded secrets in main.py
    main_py = Path("backend/main.py")
    if main_py.exists():
        with open(main_py) as f:
            main_content = f.read()
        
        dangerous_patterns = [
            'password =',
            'PASSWORD =',
            'secret =',
            'SECRET =',
            'api_key =',
            'API_KEY ='
        ]
        
        found_patterns = [p for p in dangerous_patterns if p in main_content]
        checker.check(
            "No Hardcoded Secrets",
            len(found_patterns) == 0,
            "No hardcoded secrets found in main.py" if len(found_patterns) == 0
            else f"Potential hardcoded secrets found: {found_patterns}",
            "warning"
        )


def check_authentication(checker: SecurityChecker):
    """Check authentication configuration"""
    checker.section("3. AUTHENTICATION SECURITY")
    
    # Check if enhanced_auth.py exists
    auth_file = Path("backend/enhanced_auth.py")
    checker.check(
        "Auth Module Exists",
        auth_file.exists(),
        "enhanced_auth.py found" if auth_file.exists()
        else "enhanced_auth.py not found!",
        "error"
    )
    
    if auth_file.exists():
        with open(auth_file) as f:
            auth_content = f.read()
        
        # Check for bcrypt/passlib
        checker.check(
            "Password Hashing",
            'passlib' in auth_content or 'bcrypt' in auth_content,
            "Password hashing library (passlib/bcrypt) is used",
            "error"
        )
        
        # Check for JWT
        checker.check(
            "JWT Implementation",
            'jwt' in auth_content.lower(),
            "JWT token implementation found",
            "error"
        )
        
        # Check for token expiration
        checker.check(
            "Token Expiration",
            'expire' in auth_content.lower() or 'expir' in auth_content.lower(),
            "Token expiration is implemented",
            "error"
        )


def check_dependencies(checker: SecurityChecker):
    """Check dependencies and security vulnerabilities"""
    checker.section("4. DEPENDENCIES")
    
    requirements_file = Path("backend/requirements.txt")
    checker.check(
        "Requirements File",
        requirements_file.exists(),
        "requirements.txt found" if requirements_file.exists()
        else "requirements.txt not found!",
        "warning"
    )
    
    if requirements_file.exists():
        with open(requirements_file) as f:
            requirements = f.read()
        
        # Check for security libraries
        checker.check(
            "passlib",
            'passlib' in requirements.lower(),
            "passlib (password hashing) is included",
            "error"
        )
        
        checker.check(
            "python-jose[cryptography]",
            'python-jose' in requirements.lower(),
            "python-jose (JWT) is included",
            "error"
        )
        
        checker.check(
            "bcrypt",
            'bcrypt' in requirements.lower() or 'passlib' in requirements.lower(),
            "bcrypt (secure hashing) is included",
            "error"
        )


def check_cors_configuration(checker: SecurityChecker):
    """Check CORS configuration"""
    checker.section("5. CORS CONFIGURATION")
    
    main_py = Path("backend/main.py")
    if main_py.exists():
        with open(main_py) as f:
            main_content = f.read()
        
        # Check if CORS is configured
        checker.check(
            "CORS Middleware",
            'CORSMiddleware' in main_content,
            "CORS middleware is configured",
            "error"
        )
        
        # Check for localhost in production
        if 'allow_origins' in main_content:
            checker.check(
                "CORS Origins",
                'localhost:3000' in main_content or 'localhost:5173' in main_content,
                "Update CORS origins for production (remove localhost)",
                "warning"
            )


def check_security_headers(checker: SecurityChecker):
    """Check security headers"""
    checker.section("6. SECURITY HEADERS")
    
    main_py = Path("backend/main.py")
    if main_py.exists():
        with open(main_py) as f:
            main_content = f.read()
        
        headers = [
            ('X-Content-Type-Options', 'nosniff'),
            ('X-Frame-Options', 'DENY'),
            ('X-XSS-Protection', '1; mode=block'),
            ('Content-Security-Policy', 'CSP'),
            ('Strict-Transport-Security', 'HSTS'),
        ]
        
        for header_name, short_name in headers:
            checker.check(
                f"{short_name} Header",
                header_name in main_content,
                f"{header_name} header is configured" if header_name in main_content
                else f"{header_name} header not found",
                "warning"
            )


def generate_secure_keys():
    """Generate secure keys for configuration"""
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}SECURE KEY GENERATION{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")
    
    import secrets
    
    jwt_secret = secrets.token_urlsafe(32)
    secret_key = secrets.token_urlsafe(32)
    
    print(f"{GREEN}Generated JWT_SECRET_KEY:{RESET}")
    print(f"{BOLD}{jwt_secret}{RESET}\n")
    
    print(f"{GREEN}Generated SECRET_KEY:{RESET}")
    print(f"{BOLD}{secret_key}{RESET}\n")
    
    print(f"{YELLOW}⚠️  Add these to your .env file:{RESET}")
    print(f"JWT_SECRET_KEY={jwt_secret}")
    print(f"SECRET_KEY={secret_key}\n")


def save_report(checker: SecurityChecker):
    """Save security check report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "checks_passed": checker.checks_passed,
        "checks_failed": checker.checks_failed,
        "checks_warning": checker.checks_warning,
        "results": checker.results
    }
    
    report_file = Path("security_check_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"{GREEN}Report saved to: {report_file}{RESET}\n")


def main():
    """Main security checker"""
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}🔒 SECURITY CONFIGURATION CHECKER{RESET}")
    print(f"{BOLD}Istanbul AI Guide - Production Deployment{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")
    
    checker = SecurityChecker()
    
    # Run all checks
    check_environment_variables(checker)
    check_file_permissions(checker)
    check_authentication(checker)
    check_dependencies(checker)
    check_cors_configuration(checker)
    check_security_headers(checker)
    
    # Show summary
    passed = checker.summary()
    
    # Offer to generate secure keys
    if checker.checks_failed > 0 or checker.checks_warning > 0:
        response = input(f"\n{YELLOW}Would you like to generate secure keys? (y/n): {RESET}")
        if response.lower() == 'y':
            generate_secure_keys()
    
    # Save report
    save_report(checker)
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Security check cancelled by user.{RESET}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}Error running security checks: {e}{RESET}\n")
        sys.exit(1)
