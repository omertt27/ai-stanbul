#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 🔌 START SSH TUNNEL TO LLM SERVER
# ═══════════════════════════════════════════════════════════════

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# RunPod Configuration
RUNPOD_HOST="194.68.245.153"
RUNPOD_PORT="22003"
SSH_KEY="$HOME/.ssh/id_ed25519"
LOCAL_PORT="8000"
REMOTE_PORT="8000"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}🔌 STARTING SSH TUNNEL TO LLM SERVER${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${YELLOW}This will create an SSH tunnel from:${NC}"
echo "  Local:  http://localhost:$LOCAL_PORT"
echo "  Remote: RunPod server port $REMOTE_PORT"
echo ""
echo -e "${YELLOW}Keep this terminal open!${NC}"
echo "Press Ctrl+C to stop the tunnel"
echo ""
echo -e "${GREEN}Testing server first...${NC}"

# Test if server is accessible via SSH
ssh -p "$RUNPOD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    root@"$RUNPOD_HOST" 'curl -s http://localhost:8000/health' > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Server is running and accessible!${NC}"
else
    echo -e "${RED}❌ Server is not responding. Is it running?${NC}"
    echo ""
    echo "Start the server first with:"
    echo "  ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_HOST"
    echo "  cd /workspace && ./start_llm_server_runpod.sh"
    exit 1
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 TUNNEL ACTIVE!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "You can now access the LLM server at:"
echo -e "${GREEN}  http://localhost:$LOCAL_PORT${NC}"
echo ""
echo "Test it in another terminal:"
echo "  curl http://localhost:$LOCAL_PORT/health"
echo ""
echo "Update your backend configuration:"
echo "  LLM_SERVER_URL=http://localhost:$LOCAL_PORT"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the tunnel${NC}"
echo ""

# Start the tunnel
ssh -L $LOCAL_PORT:localhost:$REMOTE_PORT \
    -p "$RUNPOD_PORT" \
    -i "$SSH_KEY" \
    -o StrictHostKeyChecking=no \
    -N \
    root@"$RUNPOD_HOST"
