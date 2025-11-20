#!/bin/bash

# AI Istanbul - Full System Verification Script
# This script tests all critical endpoints and reports status

echo "🚀 AI Istanbul - System Verification"
echo "===================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Backend URL
BACKEND="https://ai-stanbul.onrender.com"
FRONTEND="https://aistanbul.net"

echo "📡 Testing Backend Endpoints..."
echo ""

# Test 1: Health Check
echo -n "1. Health Check (/api/health): "
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND/api/health")
if [ "$HEALTH" = "200" ]; then
    echo -e "${GREEN}✅ PASSED${NC} (HTTP $HEALTH)"
else
    echo -e "${RED}❌ FAILED${NC} (HTTP $HEALTH)"
fi

# Test 2: Root API
echo -n "2. Root API (/api): "
ROOT=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND/api")
if [ "$ROOT" = "200" ]; then
    echo -e "${GREEN}✅ PASSED${NC} (HTTP $ROOT)"
else
    echo -e "${RED}❌ FAILED${NC} (HTTP $ROOT)"
fi

# Test 3: CORS Headers
echo -n "3. CORS Headers: "
CORS=$(curl -s -I -H "Origin: $FRONTEND" "$BACKEND/api/health" | grep -i "access-control-allow-origin")
if [ ! -z "$CORS" ]; then
    echo -e "${GREEN}✅ PASSED${NC}"
    echo "   └─ $CORS"
else
    echo -e "${RED}❌ FAILED${NC} - No CORS headers found"
fi

# Test 4: Chat Stream Endpoint
echo -n "4. Chat Stream Endpoint (/api/stream): "
STREAM=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BACKEND/api/stream" \
    -H "Content-Type: application/json" \
    -d '{"message":"test","conversation_id":"test"}')
if [ "$STREAM" = "200" ]; then
    echo -e "${GREEN}✅ PASSED${NC} (HTTP $STREAM)"
else
    echo -e "${YELLOW}⚠️  CHECK${NC} (HTTP $STREAM) - May need API key"
fi

echo ""
echo "🌐 Testing Frontend..."
echo ""

# Test 5: Frontend Accessibility
echo -n "5. Frontend ($FRONTEND): "
FRONT=$(curl -s -o /dev/null -w "%{http_code}" "$FRONTEND")
if [ "$FRONT" = "200" ]; then
    echo -e "${GREEN}✅ PASSED${NC} (HTTP $FRONT)"
else
    echo -e "${RED}❌ FAILED${NC} (HTTP $FRONT)"
fi

# Test 6: WWW Redirect
echo -n "6. WWW Redirect: "
WWW=$(curl -s -o /dev/null -w "%{http_code}" "https://www.aistanbul.net")
if [ "$WWW" = "200" ] || [ "$WWW" = "308" ]; then
    echo -e "${GREEN}✅ PASSED${NC} (HTTP $WWW)"
else
    echo -e "${RED}❌ FAILED${NC} (HTTP $WWW)"
fi

echo ""
echo "🔒 Testing SSL Certificates..."
echo ""

# Test 7: Backend SSL
echo -n "7. Backend SSL: "
BACKEND_SSL=$(curl -s -o /dev/null -w "%{ssl_verify_result}" "$BACKEND/api/health")
if [ "$BACKEND_SSL" = "0" ]; then
    echo -e "${GREEN}✅ VALID${NC}"
else
    echo -e "${RED}❌ INVALID${NC}"
fi

# Test 8: Frontend SSL
echo -n "8. Frontend SSL: "
FRONTEND_SSL=$(curl -s -o /dev/null -w "%{ssl_verify_result}" "$FRONTEND")
if [ "$FRONTEND_SSL" = "0" ]; then
    echo -e "${GREEN}✅ VALID${NC}"
else
    echo -e "${RED}❌ INVALID${NC}"
fi

echo ""
echo "📊 DNS Status..."
echo ""

# Test 9: DNS Resolution
echo -n "9. aistanbul.net DNS: "
DNS_ROOT=$(dig +short aistanbul.net | head -1)
if [ ! -z "$DNS_ROOT" ]; then
    echo -e "${GREEN}✅ RESOLVED${NC} → $DNS_ROOT"
else
    echo -e "${RED}❌ NOT RESOLVED${NC}"
fi

echo -n "10. www.aistanbul.net DNS: "
DNS_WWW=$(dig +short www.aistanbul.net | head -1)
if [ ! -z "$DNS_WWW" ]; then
    echo -e "${GREEN}✅ RESOLVED${NC} → $DNS_WWW"
else
    echo -e "${RED}❌ NOT RESOLVED${NC}"
fi

echo -n "11. api.aistanbul.net DNS: "
DNS_API=$(dig +short api.aistanbul.net | head -1)
if [ ! -z "$DNS_API" ]; then
    echo -e "${GREEN}✅ RESOLVED${NC} → $DNS_API"
else
    echo -e "${YELLOW}⚠️  NOT CONFIGURED${NC}"
fi

echo ""
echo "===================================="
echo "📋 Summary"
echo "===================================="
echo ""
echo "Backend Health: $(curl -s $BACKEND/api/health | python3 -m json.tool 2>/dev/null || echo 'Unable to fetch')"
echo ""
echo "🔗 URLs:"
echo "   Frontend: $FRONTEND"
echo "   Backend:  $BACKEND"
echo "   API Docs: $BACKEND/docs"
echo ""
echo "✅ Verification complete!"
echo ""
echo "Next steps:"
echo "1. If all tests pass, visit $FRONTEND and test the chat"
echo "2. Check browser console for any errors"
echo "3. Verify API calls don't have double /ai/ai/ paths"
echo ""
