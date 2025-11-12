#!/bin/bash

# Frontend Integration Verification Script
# Tests that frontend is properly configured to use Pure LLM backend

echo "🔍 Frontend Integration Verification"
echo "====================================="
echo ""

# Check 1: .env file
echo "✅ Check 1: Frontend .env configuration"
if grep -q "VITE_API_URL=http://localhost:8001" frontend/.env; then
    echo "   ✅ PASS: .env points to port 8001"
else
    echo "   ❌ FAIL: .env does not point to port 8001"
    exit 1
fi
echo ""

# Check 2: api.js file
echo "✅ Check 2: API endpoint configuration"
if grep -q "http://localhost:8001" frontend/src/api/api.js; then
    echo "   ✅ PASS: api.js uses port 8001"
else
    echo "   ❌ FAIL: api.js does not use port 8001"
    exit 1
fi

if grep -q "/api/chat" frontend/src/api/api.js; then
    echo "   ✅ PASS: api.js uses /api/chat endpoint"
else
    echo "   ❌ FAIL: api.js does not use /api/chat endpoint"
    exit 1
fi
echo ""

# Check 3: locationApi.js file
echo "✅ Check 3: Location API configuration"
if grep -q "http://localhost:8001" frontend/src/services/locationApi.js; then
    echo "   ✅ PASS: locationApi.js uses port 8001"
else
    echo "   ❌ FAIL: locationApi.js does not use port 8001"
    exit 1
fi
echo ""

# Check 4: Backend health
echo "✅ Check 4: Backend health check"
if curl -s http://localhost:8001/health | grep -q "healthy"; then
    echo "   ✅ PASS: Backend is healthy"
else
    echo "   ❌ FAIL: Backend is not responding"
    echo "   Run: python3 backend/main_pure_llm.py"
    exit 1
fi
echo ""

# Check 5: Pure LLM status
echo "✅ Check 5: Pure LLM status"
if curl -s http://localhost:8001/api/chat/status | grep -q "Llama 3.1 8B"; then
    echo "   ✅ PASS: Pure LLM is available"
else
    echo "   ❌ FAIL: Pure LLM is not available"
    exit 1
fi
echo ""

echo "====================================="
echo "🎉 ALL CHECKS PASSED!"
echo "====================================="
echo ""
echo "Frontend is now configured to use Pure LLM backend!"
echo ""
echo "Next steps:"
echo "1. Start frontend: cd frontend && npm run dev"
echo "2. Open browser: http://localhost:5173"
echo "3. Try asking: 'What is Hagia Sophia?'"
echo ""
