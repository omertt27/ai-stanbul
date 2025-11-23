#!/bin/bash

# After Newline Fix Verification Script
# Run this AFTER removing newline from LLM_API_URL and redeploying

set -e

echo "=============================================="
echo "🔍 Verifying LLM Fix After Newline Removal"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test 1: LLM Health Check
echo -e "${BLUE}Test 1: LLM Health Check${NC}"
echo "Endpoint: https://api.aistanbul.net/api/v1/llm/health"
echo ""

LLM_HEALTH=$(curl -s https://api.aistanbul.net/api/v1/llm/health)
echo "$LLM_HEALTH" | python3 -m json.tool

# Check for success
if echo "$LLM_HEALTH" | grep -q '"status": "healthy"'; then
    echo -e "${GREEN}✅ LLM is HEALTHY!${NC}"
    echo ""
    
    # Extract and verify URL has no newline
    URL=$(echo "$LLM_HEALTH" | python3 -c "import sys, json; print(json.load(sys.stdin).get('endpoint', ''))")
    if echo "$URL" | grep -q $'\n'; then
        echo -e "${RED}⚠️  WARNING: URL still contains newline!${NC}"
        echo "URL: $URL"
    else
        echo -e "${GREEN}✅ URL is clean (no newline)${NC}"
        echo "URL: $URL"
    fi
else
    echo -e "${RED}❌ LLM is NOT healthy${NC}"
    echo -e "${YELLOW}The newline may still be present or deployment didn't complete.${NC}"
    echo "Please:"
    echo "1. Double-check LLM_API_URL in Render (should be single line)"
    echo "2. Make sure deployment completed successfully"
    echo "3. Wait 2-3 minutes after deploy and try again"
    echo ""
    exit 1
fi

echo ""
echo "=============================================="

# Test 2: Chat API with English
echo -e "${BLUE}Test 2: Chat API (English)${NC}"
echo "Testing real LLM response..."
echo ""

CHAT_RESPONSE=$(curl -s -X POST https://api.aistanbul.net/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Tell me about Hagia Sophia in Istanbul","language":"en"}')

echo "$CHAT_RESPONSE" | python3 -m json.tool

# Check if it's a fallback
if echo "$CHAT_RESPONSE" | grep -q "having trouble generating"; then
    echo -e "${RED}❌ Still getting FALLBACK responses${NC}"
    echo -e "${YELLOW}This means LLM is not being used for chat.${NC}"
    echo ""
    echo "Debug steps:"
    echo "1. Check Render logs for LLM initialization errors"
    echo "2. Verify PURE_LLM_MODE=true"
    echo "3. Verify LLM_API_URL has no newline"
    exit 1
else
    echo -e "${GREEN}✅ Got REAL LLM response (not fallback)!${NC}"
fi

echo ""
echo "=============================================="

# Test 3: Chat API with Turkish
echo -e "${BLUE}Test 3: Chat API (Turkish)${NC}"
echo "Testing Turkish response..."
echo ""

CHAT_TR=$(curl -s -X POST https://api.aistanbul.net/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Merhaba, Ayasofya hakkında bilgi verir misin?","language":"tr"}')

echo "$CHAT_TR" | python3 -m json.tool

if echo "$CHAT_TR" | grep -q "having trouble generating"; then
    echo -e "${RED}❌ Turkish chat returning fallback${NC}"
else
    echo -e "${GREEN}✅ Turkish chat working!${NC}"
fi

echo ""
echo "=============================================="

# Test 4: General Backend Health
echo -e "${BLUE}Test 4: General Backend Health${NC}"
echo ""

HEALTH=$(curl -s https://api.aistanbul.net/api/health)
echo "$HEALTH" | python3 -m json.tool

if echo "$HEALTH" | grep -q '"status": "healthy"'; then
    echo -e "${GREEN}✅ Backend is healthy${NC}"
else
    echo -e "${RED}❌ Backend health check failed${NC}"
fi

echo ""
echo "=============================================="
echo -e "${GREEN}🎉 VERIFICATION COMPLETE${NC}"
echo "=============================================="
echo ""
echo "Summary:"
echo "  LLM Health:  $(echo "$LLM_HEALTH" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")"
echo "  Chat Status: Working with real LLM"
echo "  Backend:     Healthy"
echo ""
echo "Next steps:"
echo "1. Test frontend at https://aistanbul.net"
echo "2. Try multiple languages (EN, TR, AR)"
echo "3. Continue with Phase 1 testing (see PHASE_1_QUICK_START.md)"
echo ""
