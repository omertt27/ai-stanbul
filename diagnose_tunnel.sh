#!/bin/bash

echo "🔍 Cloudflare Tunnel Diagnostic"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

TUNNEL_ID="3c9f3076-300f-4a61-b923-cf7be81e2919"

echo "1️⃣  Checking Tunnel Process Status"
echo "-----------------------------------"
if pgrep -f cloudflared > /dev/null; then
    echo -e "${GREEN}✅ Tunnel process is running${NC}"
    echo "PID(s): $(pgrep -f cloudflared)"
    echo ""
    echo "Process details:"
    ps aux | grep cloudflared | grep -v grep
else
    echo -e "${RED}❌ Tunnel process is NOT running${NC}"
fi

echo ""
echo "2️⃣  Checking Local Service (localhost:8000)"
echo "-------------------------------------------"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Local service is responding${NC}"
    echo "Response:"
    curl -s http://localhost:8000/health | jq . 2>/dev/null || curl -s http://localhost:8000/health
else
    echo -e "${RED}❌ Local service is NOT responding${NC}"
    echo "The tunnel can't connect because localhost:8000 is not running!"
fi

echo ""
echo "3️⃣  Checking Credentials"
echo "------------------------"
CREDS_FILE=~/.cloudflared/${TUNNEL_ID}.json
if [ -f "$CREDS_FILE" ]; then
    echo -e "${GREEN}✅ Credentials file exists${NC}"
    echo "Location: $CREDS_FILE"
    echo "Content:"
    cat "$CREDS_FILE"
    echo ""
    
    # Validate JSON
    if python3 -m json.tool "$CREDS_FILE" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ JSON is valid${NC}"
    else
        echo -e "${RED}❌ JSON is INVALID${NC}"
    fi
else
    echo -e "${RED}❌ Credentials file NOT found${NC}"
fi

echo ""
echo "4️⃣  Checking Full Tunnel Logs"
echo "-----------------------------"
LOG_FILE=/workspace/logs/cloudflare-tunnel.log
if [ -f "$LOG_FILE" ]; then
    echo "Last 50 lines of logs:"
    echo "======================"
    tail -n 50 "$LOG_FILE"
    echo ""
    echo "======================"
    
    # Check for specific error patterns
    echo ""
    echo "Error Analysis:"
    if grep -q "control stream encountered a failure" "$LOG_FILE"; then
        echo -e "${YELLOW}⚠️  Control stream failures detected${NC}"
        echo "   This usually means the tunnel can't connect to the backend service"
    fi
    
    if grep -q "error dialing origin" "$LOG_FILE"; then
        echo -e "${YELLOW}⚠️  Origin dialing errors detected${NC}"
        echo "   The tunnel can't reach localhost:8000"
    fi
    
    if grep -q "Failed to serve" "$LOG_FILE"; then
        echo -e "${YELLOW}⚠️  Service failures detected${NC}"
    fi
    
    if grep -q "Registered tunnel connection" "$LOG_FILE"; then
        echo -e "${GREEN}✅ Tunnel successfully registered with Cloudflare${NC}"
    else
        echo -e "${RED}❌ Tunnel has NOT registered successfully${NC}"
    fi
else
    echo -e "${RED}❌ Log file NOT found${NC}"
fi

echo ""
echo "5️⃣  Checking Network Connectivity"
echo "---------------------------------"
if curl -s -o /dev/null -w "%{http_code}" https://www.cloudflare.com > /dev/null; then
    echo -e "${GREEN}✅ Can reach Cloudflare${NC}"
else
    echo -e "${RED}❌ Cannot reach Cloudflare${NC}"
fi

echo ""
echo "6️⃣  Checking Config File"
echo "------------------------"
CONFIG_FILE=~/.cloudflared/config.yml
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${GREEN}✅ Config file exists${NC}"
    echo "Content:"
    cat "$CONFIG_FILE"
else
    echo -e "${YELLOW}⚠️  No config.yml (optional when using --url flag)${NC}"
fi

echo ""
echo "7️⃣  Testing Tunnel Endpoint"
echo "---------------------------"
TUNNEL_URL="https://${TUNNEL_ID}.cfargotunnel.com/health"
echo "Testing: $TUNNEL_URL"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$TUNNEL_URL" 2>&1)
if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✅ Tunnel is accessible from internet!${NC}"
    echo "Response:"
    curl -s "$TUNNEL_URL"
else
    echo -e "${RED}❌ Tunnel is NOT accessible (HTTP $HTTP_CODE)${NC}"
    echo "This is expected if no public hostname is configured"
fi

echo ""
echo "================================"
echo "📋 Summary & Recommendations"
echo "================================"
echo ""

# Check if service is the problem
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${RED}🔴 CRITICAL: Backend service not running${NC}"
    echo ""
    echo "The tunnel is failing because localhost:8000 is not responding."
    echo ""
    echo "Solutions:"
    echo "  1. Start your backend service first"
    echo "  2. Verify it's running on port 8000"
    echo "  3. Test with: curl http://localhost:8000/health"
    echo "  4. Then restart the tunnel"
    echo ""
elif pgrep -f cloudflared > /dev/null && grep -q "control stream encountered a failure" "$LOG_FILE" 2>/dev/null; then
    echo -e "${YELLOW}🟡 Tunnel process running but failing to serve${NC}"
    echo ""
    echo "The tunnel starts but can't maintain connection to backend."
    echo ""
    echo "This might be because:"
    echo "  1. Backend service started AFTER tunnel"
    echo "  2. Backend is slow to respond"
    echo "  3. Firewall/network issues"
    echo ""
    echo "Try restarting the tunnel now that backend is confirmed working."
else
    echo -e "${GREEN}🟢 Everything looks good!${NC}"
    echo ""
    echo "If tunnel still not working, check Cloudflare dashboard:"
    echo "  https://one.dash.cloudflare.com/"
fi

echo ""
