#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 🚀 QUICK START LLM SERVER - One Command
# ═══════════════════════════════════════════════════════════════

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# RunPod Configuration
RUNPOD_HOST="194.68.245.153"
RUNPOD_PORT="22003"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}🚀 STARTING LLM SERVER ON RUNPOD${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Step 1: Check SSH connection
echo -e "${BLUE}Step 1: Checking SSH connection...${NC}"
ssh -p "$RUNPOD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    root@"$RUNPOD_HOST" 'echo "Connected!"' > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ SSH connection successful!${NC}"
else
    echo -e "${RED}❌ Cannot connect to RunPod${NC}"
    echo "Check your SSH key and connection details"
    exit 1
fi

echo ""

# Step 2: Check if files exist
echo -e "${BLUE}Step 2: Checking server files...${NC}"
FILE_CHECK=$(ssh -p "$RUNPOD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    root@"$RUNPOD_HOST" 'test -f /workspace/llm_server.py && echo "exists"')

if [ "$FILE_CHECK" = "exists" ]; then
    echo -e "${GREEN}✅ Server files found!${NC}"
else
    echo -e "${YELLOW}⚠️  Server files not found. Uploading...${NC}"
    ./upload_to_runpod.sh
fi

echo ""

# Step 3: Start the server
echo -e "${BLUE}Step 3: Starting server...${NC}"
ssh -p "$RUNPOD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    root@"$RUNPOD_HOST" << 'ENDSSH'
cd /workspace
./start_llm_server_runpod.sh
ENDSSH

echo ""

# Step 4: Wait for server to be ready
echo -e "${BLUE}Step 4: Waiting for server to be ready (30 seconds)...${NC}"
sleep 30

echo ""

# Step 5: Test the server
echo -e "${BLUE}Step 5: Testing server...${NC}"
HEALTH_CHECK=$(ssh -p "$RUNPOD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    root@"$RUNPOD_HOST" 'curl -s http://localhost:8000/health' 2>/dev/null)

if [ -n "$HEALTH_CHECK" ]; then
    echo -e "${GREEN}✅ Server is running!${NC}"
    echo ""
    echo "$HEALTH_CHECK" | python3 -m json.tool
else
    echo -e "${YELLOW}⚠️  Server started but still loading...${NC}"
    echo "Check logs with:"
    echo "  ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_HOST 'tail -f /workspace/logs/llm_server.log'"
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🎉 SERVER STARTUP COMPLETE!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Start SSH tunnel (in a new terminal):"
echo "   ./start_tunnel.sh"
echo ""
echo "2. Test the server (in another terminal):"
echo "   curl http://localhost:8000/health"
echo ""
echo "3. Update your backend:"
echo "   LLM_SERVER_URL=http://localhost:8000"
echo ""
