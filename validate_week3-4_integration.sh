#!/bin/bash

# Week 3-4 Vercel + Render Integration Validator
# Validates that all Week 3-4 components are properly set up

set -e

echo "=================================================="
echo "🔍 Week 3-4 Integration Validator"
echo "   Platform: Vercel + Render"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0

# Helper functions
check_pass() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASSED++))
}

check_fail() {
    echo -e "${RED}❌ $1${NC}"
    ((FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo "📦 Step 1: Checking Backend Files"
echo "----------------------------------"

# Check backend services
if [ -f "backend/services/redis_cache.py" ]; then
    check_pass "Redis cache service exists"
else
    check_fail "Redis cache service missing (backend/services/redis_cache.py)"
fi

if [ -f "backend/services/realtime_feedback_loop.py" ]; then
    check_pass "Feedback loop service exists"
else
    check_fail "Feedback loop service missing"
fi

if [ -f "backend/services/recommendation_ab_testing.py" ]; then
    check_pass "A/B testing service exists"
else
    check_fail "A/B testing service missing"
fi

# Check API routes
if [ -f "backend/api/recommendation_routes.py" ]; then
    check_pass "Recommendation API routes exist"
else
    check_fail "Recommendation routes missing (backend/api/recommendation_routes.py)"
fi

if [ -f "backend/api/ab_testing_routes.py" ]; then
    check_pass "A/B testing API routes exist"
else
    check_fail "A/B testing routes missing"
fi

if [ -f "backend/api/monitoring_routes.py" ]; then
    check_pass "Monitoring API routes exist"
else
    check_fail "Monitoring routes missing"
fi

echo ""
echo "🧪 Step 2: Checking Test Suite"
echo "-------------------------------"

if [ -f "test_week3-4_production_readiness.py" ]; then
    check_pass "Production readiness tests exist"
    
    # Try to run tests (optional)
    read -p "Run tests now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Running tests..."
        if python test_week3-4_production_readiness.py; then
            check_pass "All tests passed"
        else
            check_fail "Some tests failed - check output above"
        fi
    fi
else
    check_fail "Test suite missing (test_week3-4_production_readiness.py)"
fi

echo ""
echo "📚 Step 3: Checking Documentation"
echo "----------------------------------"

if [ -f "WEEK_3-4_VERCEL_RENDER_GUIDE.md" ]; then
    check_pass "Deployment guide exists"
else
    check_fail "Deployment guide missing"
fi

if [ -f "FRONTEND_TRACKING_INTEGRATION.md" ]; then
    check_pass "Frontend integration guide exists"
else
    check_fail "Frontend guide missing"
fi

if [ -f "WEEK_3-4_DEPLOYMENT_CHECKLIST.md" ]; then
    check_pass "Deployment checklist exists"
else
    check_fail "Deployment checklist missing"
fi

echo ""
echo "🔧 Step 4: Checking Environment Variables"
echo "------------------------------------------"

# Check for .env file
if [ -f ".env" ]; then
    check_pass ".env file exists"
    
    # Check key variables
    if grep -q "REDIS_URL" .env; then
        check_pass "REDIS_URL is defined"
    else
        check_warn "REDIS_URL not in .env (needed for Render deployment)"
    fi
    
    if grep -q "DATABASE_URL" .env; then
        check_pass "DATABASE_URL is defined"
    else
        check_warn "DATABASE_URL not in .env"
    fi
else
    check_warn ".env file not found (create for local testing)"
fi

echo ""
echo "🌐 Step 5: Checking Backend Integration"
echo "----------------------------------------"

# Check if main.py includes the routers
if grep -q "from backend.api.recommendation_routes import router" backend/main.py 2>/dev/null; then
    check_pass "Recommendation routes imported in main.py"
else
    check_fail "Recommendation routes NOT imported in main.py"
fi

if grep -q "app.include_router" backend/main.py 2>/dev/null; then
    check_pass "Router registration found in main.py"
else
    check_fail "Router registration missing in main.py"
fi

echo ""
echo "🚀 Step 6: Backend Startup Check (Optional)"
echo "--------------------------------------------"

read -p "Test backend startup? This will start the server briefly. (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting backend..."
    
    # Try to start backend for 3 seconds
    cd backend 2>/dev/null || true
    timeout 3s python -m uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/backend_startup.log 2>&1 &
    BACKEND_PID=$!
    
    sleep 2
    
    # Check if process is still running
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        check_pass "Backend starts without errors"
        kill $BACKEND_PID 2>/dev/null || true
    else
        check_fail "Backend failed to start - check logs"
        echo "Last 10 lines of startup log:"
        tail -n 10 /tmp/backend_startup.log || true
    fi
    
    cd .. 2>/dev/null || true
fi

echo ""
echo "=================================================="
echo "📊 Validation Summary"
echo "=================================================="
echo ""
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Week 3-4 is ready for deployment.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Review WEEK_3-4_DEPLOYMENT_CHECKLIST.md"
    echo "2. Set environment variables in Render"
    echo "3. Deploy backend: git push origin main"
    echo "4. Deploy frontend to Vercel"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Some checks failed. Please fix the issues above.${NC}"
    echo ""
    echo "Common fixes:"
    echo "- Run: python setup_week3-4.sh (to create missing files)"
    echo "- Verify backend/api/ directory exists"
    echo "- Check import paths in backend/main.py"
    echo ""
    exit 1
fi
