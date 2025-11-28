#!/bin/bash

# 🔧 Fix SSH Tunnel to RunPod vLLM

echo "🔍 Checking tunnel status..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Kill existing tunnel
echo -e "${YELLOW}Killing existing SSH tunnel...${NC}"
pkill -f "ssh.*pvj233wwhiu6j3.*8000" 2>/dev/null || true
sleep 2

# Test if port is free
if lsof -i :8000 >/dev/null 2>&1; then
    echo -e "${RED}Port 8000 is still in use!${NC}"
    echo "What's using it:"
    lsof -i :8000
    echo ""
    echo "Kill it with: sudo lsof -ti:8000 | xargs kill -9"
    exit 1
fi

echo -e "${GREEN}Port 8000 is free${NC}"
echo ""

# Create new tunnel
echo -e "${BLUE}Creating new SSH tunnel...${NC}"
ssh -f -N -L 8000:localhost:8000 \
  pvj233wwhiu6j3-64411542@ssh.runpod.io \
  -i ~/.ssh/id_ed25519 \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -o ConnectTimeout=10

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to create SSH tunnel!${NC}"
    echo ""
    echo "Possible issues:"
    echo "  1. RunPod pod is stopped/terminated"
    echo "  2. SSH key issue"
    echo "  3. RunPod SSH service down"
    echo ""
    echo "Check your pod at: https://www.runpod.io/console/pods"
    exit 1
fi

echo -e "${GREEN}✅ SSH tunnel created${NC}"
echo ""

# Wait for tunnel to establish
echo "Waiting 3 seconds for tunnel to establish..."
sleep 3

# Test the tunnel
echo -e "${BLUE}Testing vLLM connection...${NC}"
RESPONSE=$(curl -s -m 5 http://localhost:8000/health 2>&1)

if echo "$RESPONSE" | grep -q "model_name\|vllm\|Healthy"; then
    echo -e "${GREEN}✅ SUCCESS! vLLM is responding!${NC}"
    echo ""
    echo "Response:"
    echo "$RESPONSE" | head -10
    echo ""
    echo -e "${GREEN}🎉 Tunnel is working! You can now proceed with deployment.${NC}"
    exit 0
else
    echo -e "${RED}❌ vLLM is not responding through the tunnel${NC}"
    echo ""
    echo "Response received:"
    echo "$RESPONSE"
    echo ""
    echo -e "${YELLOW}⚠️  This means vLLM is likely not running on RunPod!${NC}"
    echo ""
    echo "To fix:"
    echo "  1. Open RunPod web terminal: https://www.runpod.io/console/pods"
    echo "  2. Run this command:"
    echo ""
    echo "     pkill -9 -f vllm && python3 -m vllm.entrypoints.openai.api_server \\"
    echo "       --model meta-llama/Meta-Llama-3.1-8B-Instruct \\"
    echo "       --port 8000 \\"
    echo "       --host 0.0.0.0 \\"
    echo "       --dtype auto \\"
    echo "       --max-model-len 4096 \\"
    echo "       --gpu-memory-utilization 0.9 \\"
    echo "       > /root/vllm.log 2>&1 &"
    echo ""
    echo "  3. Wait 60 seconds for model to load"
    echo "  4. Run this script again: ./fix_tunnel.sh"
    exit 1
fi
