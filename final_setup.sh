#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 🚀 FINAL SETUP - Works with Ubuntu 24.04 restrictions
# ═══════════════════════════════════════════════════════════════

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}🚀 COMPLETE LLM SERVER SETUP (FIXED)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Step 1: Install dependencies with --break-system-packages
echo -e "${BLUE}Step 1: Installing Python dependencies...${NC}"
ssh -p 22003 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@194.68.245.153 << 'ENDSSH'
echo "📦 Installing dependencies with --break-system-packages..."
pip install --break-system-packages fastapi uvicorn[standard] transformers torch accelerate bitsandbytes pydantic requests
echo "✅ Dependencies installed!"
ENDSSH

echo -e "${GREEN}✅ Dependencies installed!${NC}"
echo ""

# Step 2: Start server with nohup
echo -e "${BLUE}Step 2: Starting server with nohup...${NC}"
ssh -p 22003 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@194.68.245.153 << 'ENDSSH'
cd /workspace

# Kill any existing server
pkill -f "llm_server.py" 2>/dev/null || true
sleep 2

# Create logs directory
mkdir -p /workspace/logs

# Start server with nohup
echo "🚀 Starting server..."
nohup python /workspace/llm_server.py > /workspace/logs/llm_server.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > /workspace/llm_server.pid
disown 2>/dev/null || true

echo "✅ Server started with PID: $SERVER_PID"
echo "⏳ Model is loading (30-60 seconds)..."
ENDSSH

echo -e "${GREEN}✅ Server is starting!${NC}"
echo ""

# Step 3: Wait for initialization
echo -e "${BLUE}Step 3: Waiting 60 seconds for model to load...${NC}"
for i in {1..12}; do
    echo -n "."
    sleep 5
done
echo ""
echo ""

# Step 4: Test server
echo -e "${BLUE}Step 4: Testing server...${NC}"
HEALTH=$(ssh -p 22003 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@194.68.245.153 \
    'curl -s http://localhost:8000/health' 2>/dev/null)

if [ -n "$HEALTH" ] && echo "$HEALTH" | grep -q "healthy"; then
    echo -e "${GREEN}✅ Server is running and healthy!${NC}"
    echo ""
    echo "$HEALTH" | python3 -m json.tool
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🎉 SUCCESS! LLM SERVER IS READY!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "📊 Server is running with nohup (persists after disconnect)"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Start SSH tunnel (new terminal):"
    echo "   ./start_tunnel.sh"
    echo ""
    echo "2. Test locally (after tunnel):"
    echo "   curl http://localhost:8000/health"
    echo ""
    echo "3. Test completion:"
    echo '   curl -X POST http://localhost:8000/v1/completions \'
    echo '     -H "Content-Type: application/json" \'
    echo '     -d '"'"'{"prompt": "Istanbul is", "max_tokens": 30}'"'"
    echo ""
    echo "4. Update backend:"
    echo "   LLM_SERVER_URL=http://localhost:8000"
    echo ""
else
    echo -e "${YELLOW}⚠️  Server is still loading model...${NC}"
    echo ""
    echo "This is normal! The model takes time to load."
    echo ""
    echo "Wait 1-2 more minutes, then test:"
    echo "   ./test_llm_server.sh"
    echo ""
    echo "Or check logs:"
    echo "   ssh -p 22003 -i ~/.ssh/id_ed25519 root@194.68.245.153 'tail -f /workspace/logs/llm_server.log'"
    echo ""
    echo "Check process:"
    echo "   ssh -p 22003 -i ~/.ssh/id_ed25519 root@194.68.245.153 'ps aux | grep llm_server.py'"
fi

echo ""
