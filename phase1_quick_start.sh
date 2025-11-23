#!/bin/bash

# Phase 1: Quick Start Automation Script
# This script automates the setup and verification of Phase 1 deployment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration (override with environment variables)
BACKEND_URL="${BACKEND_URL:-https://api.aistanbul.net}"
FRONTEND_URL="${FRONTEND_URL:-https://aistanbul.net}"
LLM_API_URL="${LLM_API_URL:-}"

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   🚀 Phase 1: Quick Start Automation Script                  ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}   $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Check prerequisites
print_header "📋 Checking Prerequisites"

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    print_success "Python 3 is installed: $(python3 --version)"
else
    print_error "Python 3 is not installed!"
    exit 1
fi

# Check if required packages are installed
print_info "Checking required Python packages..."
if python3 -c "import requests, colorama" 2>/dev/null; then
    print_success "Required Python packages are installed"
else
    print_warning "Installing required Python packages..."
    pip3 install requests colorama python-dotenv
fi

# Check if curl is available
if command -v curl &> /dev/null; then
    print_success "curl is installed"
else
    print_error "curl is not installed!"
    exit 1
fi

# Display configuration
print_header "⚙️  Configuration"
echo "Backend URL:  $BACKEND_URL"
echo "Frontend URL: $FRONTEND_URL"
echo "LLM API URL:  ${LLM_API_URL:-'Not set'}"
echo ""

if [ -z "$LLM_API_URL" ]; then
    print_warning "LLM_API_URL is not set. Some tests will be skipped."
    echo "To set it: export LLM_API_URL=https://your-runpod-url/v1"
fi

# Quick connectivity test
print_header "🔌 Quick Connectivity Tests"

print_info "Testing backend connectivity..."
if curl -s -f -o /dev/null -w "%{http_code}" "$BACKEND_URL/health" | grep -q "200"; then
    print_success "Backend is reachable"
else
    print_error "Backend is not reachable at $BACKEND_URL"
    print_info "Check if the backend is deployed and the URL is correct"
fi

print_info "Testing frontend connectivity..."
if curl -s -f -o /dev/null -w "%{http_code}" "$FRONTEND_URL" | grep -q "200"; then
    print_success "Frontend is reachable"
else
    print_error "Frontend is not reachable at $FRONTEND_URL"
    print_info "Check if the frontend is deployed and the URL is correct"
fi

# Run health check script
print_header "🏥 Running Health Check Suite"
print_info "This will test all critical endpoints..."
echo ""

export BACKEND_URL FRONTEND_URL LLM_API_URL
if python3 phase1_health_check.py; then
    print_success "Health check passed!"
else
    print_error "Health check failed!"
    print_info "Review the output above for specific failures"
fi

# Run multi-language tests
print_header "🌍 Running Multi-Language Tests"
print_info "Testing all 6 supported languages with comprehensive scenarios..."
echo ""

if python3 phase1_multilang_tests.py; then
    print_success "Multi-language tests passed!"
else
    print_warning "Multi-language tests had some failures"
    print_info "Review the output above for specific issues"
fi

# Summary
print_header "📊 Phase 1 Setup Summary"

# Check for test reports
if ls health_check_report_*.json 1> /dev/null 2>&1; then
    latest_health=$(ls -t health_check_report_*.json | head -1)
    print_info "Latest health check report: $latest_health"
fi

if ls multilang_test_report_*.json 1> /dev/null 2>&1; then
    latest_multilang=$(ls -t multilang_test_report_*.json | head -1)
    print_info "Latest multi-language report: $latest_multilang"
fi

echo ""
print_info "Phase 1 setup and verification complete!"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Review test reports for any failures"
echo "2. Fix any identified issues"
echo "3. Re-run tests until 100% pass rate"
echo "4. Document production URLs and credentials"
echo "5. Move to Phase 2: Modular Handler Implementation"
echo ""
echo -e "${BLUE}Documentation:${NC}"
echo "  - Environment Setup: phase1_environment_setup.md"
echo "  - Quick Start Guide: PHASE_1_QUICK_START.md"
echo "  - Full Enhancement Plan: NEW_ENHANCEMENT_PLAN_2025.md"
echo ""

print_success "🎉 Phase 1 automation complete!"
