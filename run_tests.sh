#!/bin/bash

# AI Istanbul Chatbot - Comprehensive Test Runner
# This script runs all tests with proper coverage reporting

set -e  # Exit on any error

echo "🧪 AI Istanbul Chatbot - Comprehensive Test Suite"
echo "================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    print_warning "No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
else
    print_status "Activating virtual environment..."
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        source .venv/bin/activate
    fi
fi

# Install test dependencies
print_status "Installing test dependencies..."
pip install -r tests/requirements-test.txt

# Install backend dependencies
print_status "Installing backend dependencies..."
pip install -r backend/requirements.txt

# Create necessary directories
mkdir -p htmlcov
mkdir -p test-reports

# Set environment variables for testing
export TESTING=true
export DATABASE_URL="sqlite:///test_istanbul.db"
export REDIS_URL="redis://localhost:6379/1"
export OPENAI_API_KEY="test-key-for-testing"
export ANTHROPIC_API_KEY="test-key-for-testing"
export GOOGLE_API_KEY="test-key-for-testing"

print_status "Environment variables set for testing"

# Function to run specific test categories
run_test_category() {
    local category=$1
    local description=$2
    
    print_status "Running $description..."
    
    if pytest tests/test_${category}.py -v --tb=short; then
        print_success "$description completed successfully"
        return 0
    else
        print_error "$description failed"
        return 1
    fi
}

# Function to check test coverage
check_coverage() {
    print_status "Checking test coverage..."
    
    coverage report --show-missing
    coverage html
    
    # Get coverage percentage
    COVERAGE=$(coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')
    
    if [ -n "$COVERAGE" ]; then
        if (( $(echo "$COVERAGE >= 70" | bc -l) )); then
            print_success "Coverage target met: ${COVERAGE}%"
        else
            print_warning "Coverage below target: ${COVERAGE}% (target: 70%)"
        fi
    fi
}

# Main test execution
main() {
    local failed_tests=0
    
    print_status "Starting comprehensive test suite..."
    
    # 1. Unit Tests - API Endpoints
    print_status "Phase 1: API Endpoint Tests"
    if ! run_test_category "api_endpoints" "API endpoint tests"; then
        ((failed_tests++))
    fi
    
    # 2. AI and Multilingual Tests
    print_status "Phase 2: AI and Multilingual Tests"
    if ! run_test_category "ai_multilingual" "AI and multilingual functionality tests"; then
        ((failed_tests++))
    fi
    
    # 3. GDPR Compliance Tests
    print_status "Phase 3: GDPR Compliance Tests"
    if ! run_test_category "gdpr_compliance" "GDPR compliance tests"; then
        ((failed_tests++))
    fi
    
    # 4. Performance Tests
    print_status "Phase 4: Performance Tests"
    if ! run_test_category "performance" "Performance and load tests"; then
        ((failed_tests++))
    fi
    
    # 5. Integration Tests
    print_status "Phase 5: Integration Tests"
    if ! run_test_category "integration" "End-to-end integration tests"; then
        ((failed_tests++))
    fi
    
    # 6. Full test suite with coverage
    print_status "Phase 6: Complete Test Suite with Coverage"
    if pytest tests/ --cov=backend --cov-report=html --cov-report=term-missing --cov-fail-under=70 -v; then
        print_success "Complete test suite passed"
    else
        print_error "Some tests in the complete suite failed"
        ((failed_tests++))
    fi
    
    # Generate coverage report
    check_coverage
    
    # Generate test report
    print_status "Generating test reports..."
    pytest tests/ --junit-xml=test-reports/junit.xml --html=test-reports/report.html --self-contained-html
    
    # Summary
    echo ""
    echo "🎯 Test Execution Summary"
    echo "========================"
    
    if [ $failed_tests -eq 0 ]; then
        print_success "All test phases completed successfully! ✅"
        print_success "Coverage report: htmlcov/index.html"
        print_success "Test report: test-reports/report.html"
        echo ""
        echo "🚀 Production deployment readiness: CONFIRMED"
        return 0
    else
        print_error "$failed_tests test phase(s) failed ❌"
        print_warning "Please review failures before production deployment"
        return 1
    fi
}

# Help function
show_help() {
    echo "AI Istanbul Chatbot Test Runner"
    echo ""
    echo "Usage: ./run_tests.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --api               Run only API endpoint tests"
    echo "  --ai                Run only AI and multilingual tests"
    echo "  --gdpr              Run only GDPR compliance tests"
    echo "  --performance       Run only performance tests"
    echo "  --integration       Run only integration tests"
    echo "  --coverage          Run all tests with coverage report"
    echo "  --quick             Run quick test subset"
    echo ""
    echo "Examples:"
    echo "  ./run_tests.sh                    # Run all tests"
    echo "  ./run_tests.sh --api             # Run only API tests"
    echo "  ./run_tests.sh --coverage        # Run with detailed coverage"
    echo ""
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --api)
        run_test_category "api_endpoints" "API endpoint tests"
        ;;
    --ai)
        run_test_category "ai_multilingual" "AI and multilingual functionality tests"
        ;;
    --gdpr)
        run_test_category "gdpr_compliance" "GDPR compliance tests"
        ;;
    --performance)
        run_test_category "performance" "Performance and load tests"
        ;;
    --integration)
        run_test_category "integration" "End-to-end integration tests"
        ;;
    --coverage)
        pytest tests/ --cov=backend --cov-report=html --cov-report=term-missing --cov-fail-under=70 -v
        check_coverage
        ;;
    --quick)
        print_status "Running quick test subset..."
        pytest tests/test_api_endpoints.py::TestAPIEndpoints::test_root_endpoint \
               tests/test_ai_multilingual.py::TestAIMultilingualFunctionality::test_language_detection_english \
               tests/test_gdpr_compliance.py::TestGDPRCompliance::test_consent_management \
               -v
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
