#!/bin/bash
# 🔒 Security Audit Script for AI Istanbul

set -e

echo "🔒 AI Istanbul - Security Audit"
echo "================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Installing security tools..."
    
    # Install OWASP ZAP
    if ! command -v zaproxy &> /dev/null; then
        log_warn "OWASP ZAP not found. Install from: https://www.zaproxy.org/"
    fi
    
    # Install Bandit (Python security linter)
    pip install bandit > /dev/null 2>&1 || log_warn "Bandit installation failed"
    
    # Install Safety (Python dependency checker)
    pip install safety > /dev/null 2>&1 || log_warn "Safety installation failed"
    
    # Install npm audit
    if ! command -v npm &> /dev/null; then
        log_warn "npm not found, skipping frontend security checks"
    fi
}

# Check SSL/TLS configuration
check_ssl() {
    log_info "Checking SSL/TLS configuration..."
    
    local domain=$1
    
    if [ -z "$domain" ]; then
        log_warn "No domain specified, skipping SSL check"
        return
    fi
    
    # Check SSL certificate
    if command -v openssl &> /dev/null; then
        echo | openssl s_client -connect $domain:443 -servername $domain 2>/dev/null | openssl x509 -noout -dates
        
        # Check SSL protocols
        log_info "Testing SSL protocols..."
        for protocol in ssl2 ssl3 tls1 tls1_1 tls1_2 tls1_3; do
            result=$(echo | timeout 2 openssl s_client -connect $domain:443 -$protocol 2>&1 | grep "Protocol")
            if [ -n "$result" ]; then
                echo "  $protocol: ${GREEN}Supported${NC}"
            fi
        done
    fi
}

# Check security headers
check_security_headers() {
    log_info "Checking security headers..."
    
    local url=$1
    
    if [ -z "$url" ]; then
        log_warn "No URL specified, skipping header check"
        return
    fi
    
    headers=$(curl -sI $url)
    
    # Check important security headers
    check_header "X-Frame-Options" "$headers"
    check_header "X-Content-Type-Options" "$headers"
    check_header "X-XSS-Protection" "$headers"
    check_header "Strict-Transport-Security" "$headers"
    check_header "Content-Security-Policy" "$headers"
    check_header "Referrer-Policy" "$headers"
}

check_header() {
    local header=$1
    local headers=$2
    
    if echo "$headers" | grep -qi "$header"; then
        echo -e "  ${GREEN}✅${NC} $header present"
    else
        echo -e "  ${RED}❌${NC} $header missing"
    fi
}

# Scan Python code for security issues
scan_python() {
    log_info "Scanning Python code with Bandit..."
    
    cd /Users/omer/Desktop/ai-stanbul/backend
    
    bandit -r . -f json -o bandit-report.json || true
    bandit -r . || true
    
    log_info "Bandit report saved to: backend/bandit-report.json"
}

# Check Python dependencies for vulnerabilities
check_python_dependencies() {
    log_info "Checking Python dependencies with Safety..."
    
    cd /Users/omer/Desktop/ai-stanbul/backend
    
    safety check --json > safety-report.json || true
    safety check || true
    
    log_info "Safety report saved to: backend/safety-report.json"
}

# Check frontend dependencies
check_frontend_dependencies() {
    log_info "Checking frontend dependencies with npm audit..."
    
    cd /Users/omer/Desktop/ai-stanbul/frontend
    
    npm audit --json > npm-audit.json || true
    npm audit || true
    
    log_info "npm audit report saved to: frontend/npm-audit.json"
}

# Check for exposed secrets
check_secrets() {
    log_info "Checking for exposed secrets..."
    
    cd /Users/omer/Desktop/ai-stanbul
    
    # Check for common secret patterns
    log_warn "Searching for potential secrets in code..."
    
    grep -r -i "api[_-]key" --include="*.py" --include="*.js" --include="*.jsx" . | head -10 || log_info "No API keys found in code"
    grep -r -i "secret[_-]key" --include="*.py" --include="*.js" --include="*.jsx" . | head -10 || log_info "No secret keys found in code"
    grep -r -i "password" --include="*.py" --include="*.js" --include="*.jsx" . | head -10 || log_info "No passwords found in code"
    
    # Check .env files
    if [ -f "backend/.env" ]; then
        log_warn "Found .env file - ensure it's in .gitignore"
        if grep -q "\.env" .gitignore; then
            log_info "  ✅ .env is in .gitignore"
        else
            log_error "  ❌ .env is NOT in .gitignore!"
        fi
    fi
}

# Check Docker security
check_docker() {
    log_info "Checking Docker security..."
    
    # Check if running as root
    log_info "Checking Dockerfile USER directives..."
    
    for dockerfile in docker/Dockerfile.*; do
        if [ -f "$dockerfile" ]; then
            if grep -q "^USER" "$dockerfile"; then
                echo -e "  ${GREEN}✅${NC} $dockerfile has USER directive"
            else
                echo -e "  ${YELLOW}⚠️${NC}  $dockerfile missing USER directive (running as root)"
            fi
        fi
    done
    
    # Check for latest tags
    log_info "Checking for 'latest' tags in Dockerfiles..."
    grep "FROM" docker/Dockerfile.* | grep -i "latest" && log_warn "Found 'latest' tags - pin versions instead" || log_info "  ✅ No 'latest' tags found"
}

# Check rate limiting
check_rate_limiting() {
    log_info "Checking rate limiting..."
    
    local url=$1
    
    if [ -z "$url" ]; then
        log_warn "No URL specified, skipping rate limit check"
        return
    fi
    
    # Make 20 rapid requests
    log_info "Making 20 rapid requests to test rate limiting..."
    
    local blocked=0
    for i in {1..20}; do
        status=$(curl -s -o /dev/null -w "%{http_code}" $url/api/health)
        if [ "$status" = "429" ]; then
            blocked=$((blocked + 1))
        fi
    done
    
    if [ $blocked -gt 0 ]; then
        echo -e "  ${GREEN}✅${NC} Rate limiting active ($blocked/20 blocked)"
    else
        echo -e "  ${YELLOW}⚠️${NC}  Rate limiting may not be configured"
    fi
}

# Check CORS configuration
check_cors() {
    log_info "Checking CORS configuration..."
    
    local url=$1
    
    if [ -z "$url" ]; then
        log_warn "No URL specified, skipping CORS check"
        return
    fi
    
    # Check CORS headers
    cors_headers=$(curl -sI -H "Origin: https://evil.com" $url/api/health)
    
    if echo "$cors_headers" | grep -qi "Access-Control-Allow-Origin: \*"; then
        log_error "  ❌ CORS allows all origins (*) - security risk!"
    elif echo "$cors_headers" | grep -qi "Access-Control-Allow-Origin:"; then
        log_info "  ✅ CORS configured with specific origins"
    else
        log_info "  ✅ CORS not allowing cross-origin requests"
    fi
}

# Generate security report
generate_report() {
    log_info "Generating security audit report..."
    
    cat > security-audit-report.md << EOF
# 🔒 Security Audit Report

**Audit Date:** $(date)

## Summary

### Critical Issues
- Review results above for any CRITICAL findings

### High Priority Issues
- Missing security headers
- Exposed secrets
- Vulnerable dependencies
- Rate limiting issues

### Medium Priority Issues
- Docker security improvements
- CORS configuration review
- SSL/TLS configuration

### Low Priority Issues
- Code quality improvements
- Documentation updates

## Detailed Findings

### Python Security (Bandit)
- See: backend/bandit-report.json

### Dependency Vulnerabilities (Safety)
- See: backend/safety-report.json

### Frontend Dependencies (npm audit)
- See: frontend/npm-audit.json

## Recommendations

### Immediate Actions
1. Fix any CRITICAL vulnerabilities in dependencies
2. Ensure all secrets are in .env files (not committed)
3. Implement rate limiting if not configured
4. Add missing security headers

### Short-term Actions
1. Review and update Docker configurations
2. Implement Web Application Firewall (WAF)
3. Set up security monitoring (Sentry)
4. Enable HTTPS only with HSTS

### Long-term Actions
1. Regular security audits (quarterly)
2. Penetration testing
3. Bug bounty program
4. Security training for team

## Compliance

- [ ] GDPR compliance review
- [ ] Data encryption at rest
- [ ] Data encryption in transit
- [ ] Access control audit
- [ ] Logging and monitoring

## Next Steps

1. Address all critical and high priority issues
2. Re-run security audit
3. Schedule regular security reviews
4. Document security procedures
EOF
    
    log_info "Report generated: security-audit-report.md"
}

# Main execution
main() {
    echo ""
    log_info "Starting security audit..."
    echo ""
    
    check_prerequisites
    
    # Code scanning
    scan_python
    echo ""
    
    # Dependency checks
    check_python_dependencies
    echo ""
    check_frontend_dependencies
    echo ""
    
    # Secret scanning
    check_secrets
    echo ""
    
    # Docker security
    check_docker
    echo ""
    
    # Ask for URL if doing live checks
    read -p "Enter URL for live security checks (or press Enter to skip): " url
    
    if [ -n "$url" ]; then
        check_ssl $(echo $url | sed 's|https\?://||')
        echo ""
        check_security_headers $url
        echo ""
        check_rate_limiting $url
        echo ""
        check_cors $url
        echo ""
    fi
    
    # Generate report
    generate_report
    
    log_info "🎉 Security audit complete!"
    log_info "Review the generated report: security-audit-report.md"
}

# Run main
main
