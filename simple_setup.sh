#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 🚀 SIMPLE COMPLETE SETUP - One Command
# ═══════════════════════════════════════════════════════════════

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}🚀 COMPLETE LLM SERVER SETUP${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Step 1: Upload files
echo -e "${BLUE}Step 1: Uploading files...${NC}"
scp -P 22003 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no \
    llm_server.py start_server_nohup.sh download_model.sh \
    root@194.68.245.153:/workspace/
echo -e "${GREEN}✅ Files uploaded!${NC}"
echo ""

# Step 2: Install dependencies and start server
echo -e "${BLUE}Step 2: Installing dependencies and starting server...${NC}"
ssh -p 22003 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@194.68.245.153 << 'ENDSSH'
cd /workspace

# Make scripts executable
chmod +x *.sh

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q fastapi uvicorn[standard] transformers torch accelerate bitsandbytes pydantic requests

# Kill any existing server
pkill -f "llm_server.py" 2>/dev/null || true
sleep 2

# Create logs directory
mkdir -p /workspace/logs

# Start server with nohup
echo "🚀 Starting server with nohup..."
nohup python /workspace/llm_server.py > /workspace/logs/llm_server.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > /workspace/llm_server.pid
disown 2>/dev/null || true

echo "✅ Server started with PID: $SERVER_PID"
echo "⏳ Waiting for model to load (this takes 30-60 seconds)..."
ENDSSH

echo -e "${GREEN}✅ Server is starting!${NC}"
echo ""

# Step 3: Wait for initialization
echo -e "${BLUE}Step 3: Waiting 45 seconds for model to load...${NC}"
for i in {1..9}; do
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
    echo "Next steps:"
    echo ""
    echo "1. Start SSH tunnel (new terminal):"
    echo "   ./start_tunnel.sh"
    echo ""
    echo "2. Test locally:"
    echo "   curl http://localhost:8000/health"
    echo ""
    echo "3. Update backend:"
    echo "   LLM_SERVER_URL=http://localhost:8000"
    echo ""
else
    echo -e "${YELLOW}⚠️  Server is starting but still loading model...${NC}"
    echo ""
    echo "Wait 1-2 more minutes, then test:"
    echo "   ./test_llm_server.sh"
    echo ""
    echo "Or check logs:"
    echo "   ssh -p 22003 -i ~/.ssh/id_ed25519 root@194.68.245.153 'tail -f /workspace/logs/llm_server.log'"
fi

echo ""
