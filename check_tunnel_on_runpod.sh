#!/bin/bash
# Check Tunnel Status from RunPod Terminal
# Run this script while SSH'd into your RunPod instance

echo "🔍 Checking Tunnel Status on RunPod"
echo "====================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

TUNNEL_DOMAIN="asdweq123.org"

echo "📋 System Information:"
echo "   Hostname: $(hostname)"
echo "   IP: $(hostname -I | awk '{print $1}')"
echo ""

# 1. Check if vLLM is running
echo "1️⃣ Checking if vLLM is running..."
VLLM_RUNNING=$(ps aux | grep vllm | grep -v grep)
if [ -n "$VLLM_RUNNING" ]; then
    echo -e "${GREEN}✅ vLLM is running${NC}"
    echo "   Process: $(echo $VLLM_RUNNING | awk '{print $11, $12, $13, $14, $15}')"
else
    echo -e "${RED}❌ vLLM is NOT running${NC}"
    echo "   Start with: cd /workspace && nohup python -m vllm.entrypoints.openai.api_server --port 8000 ..."
fi
echo ""

# 2. Check if port 8000 is listening
echo "2️⃣ Checking if port 8000 is listening..."
PORT_LISTENING=$(netstat -tuln 2>/dev/null | grep ":8000" || ss -tuln 2>/dev/null | grep ":8000")
if [ -n "$PORT_LISTENING" ]; then
    echo -e "${GREEN}✅ Port 8000 is listening${NC}"
    echo "   $PORT_LISTENING"
else
    echo -e "${RED}❌ Port 8000 is NOT listening${NC}"
fi
echo ""

# 3. Test vLLM locally
echo "3️⃣ Testing vLLM locally (localhost:8000)..."
LOCAL_HEALTH=$(curl -s -w "\n%{http_code}" http://localhost:8000/health 2>&1)
HTTP_CODE=$(echo "$LOCAL_HEALTH" | tail -1)
BODY=$(echo "$LOCAL_HEALTH" | head -n -1)

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✅ vLLM responds on localhost${NC}"
    echo "   Response: $BODY"
else
    echo -e "${RED}❌ vLLM not responding on localhost${NC}"
    echo "   HTTP Code: $HTTP_CODE"
fi
echo ""

# 4. Check Cloudflare Tunnel
echo "4️⃣ Checking Cloudflare Tunnel (cloudflared)..."
CLOUDFLARED_RUNNING=$(ps aux | grep cloudflared | grep -v grep)
if [ -n "$CLOUDFLARED_RUNNING" ]; then
    echo -e "${GREEN}✅ Cloudflared is running${NC}"
    echo "   Process: $(echo $CLOUDFLARED_RUNNING | awk '{print $11, $12, $13}')"
    
    # Check systemd service
    if command -v systemctl &> /dev/null; then
        SERVICE_STATUS=$(systemctl is-active cloudflared 2>/dev/null)
        if [ "$SERVICE_STATUS" = "active" ]; then
            echo -e "   ${GREEN}✅ Systemd service is active${NC}"
        else
            echo -e "   ${YELLOW}⚠️  Systemd service not active (running manually)${NC}"
        fi
    fi
    
    # Check config
    if [ -f ~/.cloudflared/config.yml ]; then
        echo -e "   ${GREEN}✅ Config file exists: ~/.cloudflared/config.yml${NC}"
        echo "   Tunnel config:"
        grep -E "tunnel:|hostname:" ~/.cloudflared/config.yml | sed 's/^/      /'
    else
        echo -e "   ${YELLOW}⚠️  Config file not found${NC}"
    fi
else
    echo -e "${RED}❌ Cloudflared is NOT running${NC}"
fi
echo ""

# 5. Check SSH Reverse Tunnel
echo "5️⃣ Checking SSH Reverse Tunnel..."
SSH_TUNNEL=$(ps aux | grep "ssh.*-R.*8000" | grep -v grep)
if [ -n "$SSH_TUNNEL" ]; then
    echo -e "${GREEN}✅ SSH reverse tunnel is running${NC}"
    echo "   Process: $(echo $SSH_TUNNEL | awk '{print $11, $12, $13, $14}')"
else
    echo -e "${YELLOW}⚠️  No SSH reverse tunnel detected${NC}"
fi
echo ""

# 6. Check ngrok
echo "6️⃣ Checking ngrok..."
NGROK_RUNNING=$(ps aux | grep ngrok | grep -v grep)
if [ -n "$NGROK_RUNNING" ]; then
    echo -e "${GREEN}✅ ngrok is running${NC}"
    echo "   Process: $(echo $NGROK_RUNNING | awk '{print $11, $12, $13}')"
    
    # Try to get ngrok URL
    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -o '"public_url":"[^"]*"' | cut -d'"' -f4)
    if [ -n "$NGROK_URL" ]; then
        echo "   Public URL: $NGROK_URL"
    fi
else
    echo -e "${YELLOW}⚠️  ngrok is NOT running${NC}"
fi
echo ""

# 7. Check outbound connectivity
echo "7️⃣ Checking outbound internet connectivity..."
OUTBOUND=$(curl -s -o /dev/null -w "%{http_code}" https://google.com --connect-timeout 3 2>&1)
if [ "$OUTBOUND" = "200" ] || [ "$OUTBOUND" = "301" ]; then
    echo -e "${GREEN}✅ Internet connectivity works${NC}"
else
    echo -e "${RED}❌ No internet connectivity${NC}"
fi
echo ""

# 8. Test tunnel domain from RunPod
echo "8️⃣ Testing tunnel domain from RunPod (${TUNNEL_DOMAIN})..."
echo "   Trying to reach https://${TUNNEL_DOMAIN}/health from here..."
TUNNEL_TEST=$(curl -s -w "\n%{http_code}" https://${TUNNEL_DOMAIN}/health --connect-timeout 5 2>&1)
TUNNEL_CODE=$(echo "$TUNNEL_TEST" | tail -1)
TUNNEL_BODY=$(echo "$TUNNEL_TEST" | head -n -1)

if [ "$TUNNEL_CODE" = "200" ]; then
    echo -e "${GREEN}✅ Tunnel is working! ${TUNNEL_DOMAIN} is accessible${NC}"
    echo "   Response: $TUNNEL_BODY"
else
    echo -e "${YELLOW}⚠️  Cannot reach ${TUNNEL_DOMAIN} from RunPod${NC}"
    echo "   HTTP Code: $TUNNEL_CODE"
    echo "   Note: This is normal if tunnel not set up yet"
fi
echo ""

# 9. Summary
echo "====================================="
echo "📊 SUMMARY"
echo "====================================="
echo ""

if [ -n "$VLLM_RUNNING" ] && [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✅ vLLM is running and healthy${NC}"
else
    echo -e "${RED}❌ vLLM needs to be started${NC}"
fi

TUNNEL_TYPE="none"
if [ -n "$CLOUDFLARED_RUNNING" ]; then
    TUNNEL_TYPE="Cloudflare Tunnel"
    echo -e "${GREEN}✅ Tunnel Type: Cloudflare Tunnel${NC}"
elif [ -n "$SSH_TUNNEL" ]; then
    TUNNEL_TYPE="SSH Reverse Tunnel"
    echo -e "${GREEN}✅ Tunnel Type: SSH Reverse Tunnel${NC}"
elif [ -n "$NGROK_RUNNING" ]; then
    TUNNEL_TYPE="ngrok"
    echo -e "${GREEN}✅ Tunnel Type: ngrok${NC}"
else
    echo -e "${YELLOW}⚠️  No tunnel detected${NC}"
    echo ""
    echo "To set up a tunnel, choose one:"
    echo ""
    echo "Option 1: Cloudflare Tunnel (Recommended)"
    echo "   wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    echo "   chmod +x cloudflared-linux-amd64"
    echo "   mv cloudflared-linux-amd64 /usr/local/bin/cloudflared"
    echo "   cloudflared tunnel login"
    echo ""
    echo "Option 2: SSH Reverse Tunnel"
    echo "   ssh -R 8000:localhost:8000 root@${TUNNEL_DOMAIN} -N -f"
    echo ""
    echo "Option 3: ngrok"
    echo "   # Download from ngrok.com"
    echo "   ngrok http 8000"
fi

echo ""
echo "====================================="
echo ""

if [ "$TUNNEL_CODE" = "200" ]; then
    echo -e "${GREEN}🎉 Everything is working!${NC}"
    echo ""
    echo "Your vLLM is accessible at:"
    echo "  - Local: http://localhost:8000"
    echo "  - Public: https://${TUNNEL_DOMAIN}"
    echo ""
    echo "Next steps:"
    echo "  1. Update .env on your Mac: LLM_API_URL=https://${TUNNEL_DOMAIN}"
    echo "  2. Update Render.com env vars: LLM_API_URL=https://${TUNNEL_DOMAIN}"
    echo "  3. Test chat from frontend"
else
    echo "Next steps:"
    if [ -z "$VLLM_RUNNING" ]; then
        echo "  1. Start vLLM (see command above)"
    fi
    if [ "$TUNNEL_TYPE" = "none" ]; then
        echo "  2. Set up tunnel (see options above)"
    fi
    echo "  3. Run this script again to verify"
fi

echo ""
