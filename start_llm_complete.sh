#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 🚀 COMPLETE LLM SERVER START - One Command Execution
# ═══════════════════════════════════════════════════════════════
# This script does EVERYTHING: upload files, SSH, and start server
# Just run: ./start_llm_complete.sh
# ═══════════════════════════════════════════════════════════════

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# RunPod Configuration
RUNPOD_HOST="194.68.245.153"
RUNPOD_PORT="22077"
RUNPOD_USER="root"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}🚀 LLM SERVER COMPLETE STARTUP${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Check SSH key
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}❌ SSH key not found: $SSH_KEY${NC}"
    echo ""
    echo "Generate one with:"
    echo "  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519"
    exit 1
fi

# Check required files
if [ ! -f "llm_server.py" ]; then
    echo -e "${RED}❌ llm_server.py not found in current directory${NC}"
    exit 1
fi

if [ ! -f "start_llm_server_runpod.sh" ]; then
    echo -e "${RED}❌ start_llm_server_runpod.sh not found in current directory${NC}"
    exit 1
fi

# Step 1: Upload files
echo -e "${YELLOW}📤 Step 1: Uploading server files to RunPod...${NC}"
echo ""

echo "Uploading llm_server.py..."
scp -P "$RUNPOD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    llm_server.py "$RUNPOD_USER@$RUNPOD_HOST:/workspace/"

echo "Uploading start_llm_server_runpod.sh..."
scp -P "$RUNPOD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    start_llm_server_runpod.sh "$RUNPOD_USER@$RUNPOD_HOST:/workspace/"

echo -e "${GREEN}✅ Files uploaded successfully!${NC}"
echo ""

# Step 2: SSH and start server
echo -e "${YELLOW}🚀 Step 2: Starting LLM server on RunPod...${NC}"
echo ""

ssh -p "$RUNPOD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "$RUNPOD_USER@$RUNPOD_HOST" << 'ENDSSH'

set -e

echo "🔧 Making startup script executable..."
chmod +x /workspace/start_llm_server_runpod.sh

echo "🚀 Starting LLM server..."
cd /workspace
./start_llm_server_runpod.sh

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ LLM SERVER STARTED SUCCESSFULLY!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📍 Server Details:"
echo "   URL: http://localhost:8001"
echo "   Health: http://localhost:8001/health"
echo "   Logs: /workspace/logs/llm_server.log"
echo ""
echo "🔍 Check server status:"
echo "   curl http://localhost:8001/health"
echo ""
echo "📝 View logs:"
echo "   tail -f /workspace/logs/llm_server.log"
echo ""
echo "🛑 Stop server:"
echo "   kill \$(cat /workspace/llm_server.pid)"
echo ""

ENDSSH

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🎉 ALL DONE! Server is running on RunPod${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Test health: ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_USER@$RUNPOD_HOST 'curl http://localhost:8001/health'"
echo "  2. View logs: ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_USER@$RUNPOD_HOST 'tail -f /workspace/logs/llm_server.log'"
echo "  3. Connect your backend to: http://$RUNPOD_HOST:8001 (expose port first)"
echo ""
