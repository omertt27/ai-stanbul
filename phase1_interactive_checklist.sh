#!/bin/bash

# Phase 1 Implementation Checklist
# Interactive guide through deployment steps

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Phase 1 Implementation Checklist${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to ask yes/no
ask_yes_no() {
    while true; do
        read -p "$1 (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Function to show status
show_status() {
    if [ "$2" = "done" ]; then
        echo -e "[${GREEN}✓${NC}] $1"
    elif [ "$2" = "todo" ]; then
        echo -e "[${YELLOW}○${NC}] $1"
    else
        echo -e "[${RED}✗${NC}] $1"
    fi
}

echo -e "${YELLOW}This script will guide you through Phase 1 deployment.${NC}"
echo ""

# Step 1: RunPod Server Check
echo -e "${BLUE}─────────────────────────────────────${NC}"
echo -e "${BLUE}Step 1: RunPod LLM Server Check${NC}"
echo -e "${BLUE}─────────────────────────────────────${NC}"
echo ""
echo "You need to SSH into your RunPod pod and verify the LLM server is running."
echo ""
echo "SSH Command:"
echo -e "${GREEN}ssh ytc61lal7ag5sy-64410fe8@ssh.runpod.io -i ~/.ssh/id_ed25519${NC}"
echo ""

if ask_yes_no "Have you SSH'd into RunPod?"; then
    echo ""
    echo "Good! Now run these commands on the RunPod pod:"
    echo ""
    echo -e "${GREEN}ps aux | grep python${NC}  # Check if server is running"
    echo -e "${GREEN}curl http://localhost:8888/health${NC}  # Test local endpoint"
    echo ""
    
    if ask_yes_no "Is the LLM server running and responding with JSON?"; then
        show_status "RunPod LLM Server Running" "done"
        SERVER_RUNNING=true
    else
        show_status "RunPod LLM Server NOT running" "fail"
        SERVER_RUNNING=false
        echo ""
        echo "To start the server, run on RunPod:"
        echo -e "${GREEN}cd /workspace${NC}"
        echo -e "${GREEN}python llm_api_server_4bit.py > server.log 2>&1 &${NC}"
        echo -e "${GREEN}sleep 20${NC}"
        echo -e "${GREEN}curl http://localhost:8888/health${NC}"
        echo ""
        if ask_yes_no "Did you start the server successfully?"; then
            show_status "RunPod LLM Server Started" "done"
            SERVER_RUNNING=true
        fi
    fi
else
    show_status "RunPod SSH Check" "todo"
    SERVER_RUNNING=false
    echo ""
    echo -e "${RED}Please SSH into RunPod first and check the server status.${NC}"
    echo "Then run this script again."
    exit 1
fi

echo ""

# Step 2: Render Backend Update
if [ "$SERVER_RUNNING" = true ]; then
    echo -e "${BLUE}─────────────────────────────────────${NC}"
    echo -e "${BLUE}Step 2: Update Render Backend${NC}"
    echo -e "${BLUE}─────────────────────────────────────${NC}"
    echo ""
    echo "Now you need to update your Render backend environment variable."
    echo ""
    echo "1. Go to: ${YELLOW}https://dashboard.render.com${NC}"
    echo "2. Select your backend service"
    echo "3. Go to Environment tab"
    echo "4. Add/Update:"
    echo ""
    echo -e "${GREEN}LLM_API_URL=https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1${NC}"
    echo ""
    echo "5. Click Save Changes"
    echo "6. Wait 3-5 minutes for redeploy"
    echo ""
    
    if ask_yes_no "Have you updated Render and waited for redeploy?"; then
        show_status "Render Backend Updated" "done"
        RENDER_UPDATED=true
    else
        show_status "Render Backend Update" "todo"
        RENDER_UPDATED=false
    fi
else
    show_status "Render Backend Update" "todo"
    echo -e "${YELLOW}Skipped - Complete Step 1 first${NC}"
    RENDER_UPDATED=false
fi

echo ""

# Step 3: Test Backend
if [ "$RENDER_UPDATED" = true ]; then
    echo -e "${BLUE}─────────────────────────────────────${NC}"
    echo -e "${BLUE}Step 3: Test Backend Health${NC}"
    echo -e "${BLUE}─────────────────────────────────────${NC}"
    echo ""
    echo "Testing backend health endpoint..."
    echo ""
    
    if response=$(curl -s https://api.aistanbul.net/health); then
        echo "Response:"
        echo "$response" | head -10
        echo ""
        if ask_yes_no "Does the response look healthy (status: healthy)?"; then
            show_status "Backend Health Check" "done"
            BACKEND_HEALTHY=true
        else
            show_status "Backend Health Check" "fail"
            BACKEND_HEALTHY=false
            echo ""
            echo "Check Render logs at: https://dashboard.render.com"
        fi
    else
        show_status "Backend Health Check" "fail"
        BACKEND_HEALTHY=false
        echo -e "${RED}Could not connect to backend${NC}"
    fi
else
    show_status "Backend Health Check" "todo"
    echo -e "${YELLOW}Skipped - Complete Step 2 first${NC}"
    BACKEND_HEALTHY=false
fi

echo ""

# Step 4: Test Frontend
if [ "$BACKEND_HEALTHY" = true ]; then
    echo -e "${BLUE}─────────────────────────────────────${NC}"
    echo -e "${BLUE}Step 4: Test Frontend Chat${NC}"
    echo -e "${BLUE}─────────────────────────────────────${NC}"
    echo ""
    echo "Now test the frontend chat interface."
    echo ""
    echo "1. Open: ${YELLOW}https://aistanbul.net${NC}"
    echo "2. Open Chrome DevTools (F12 or Cmd+Option+I)"
    echo "3. Send a test message: 'Hello, tell me about Istanbul'"
    echo "4. Check for response"
    echo ""
    
    if ask_yes_no "Did the chat return a valid response?"; then
        show_status "Frontend Chat Test" "done"
        FRONTEND_WORKING=true
    else
        show_status "Frontend Chat Test" "fail"
        FRONTEND_WORKING=false
        echo ""
        echo "Check browser console for errors"
        echo "Check Network tab for failed requests"
    fi
else
    show_status "Frontend Chat Test" "todo"
    echo -e "${YELLOW}Skipped - Complete Step 3 first${NC}"
    FRONTEND_WORKING=false
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ "$SERVER_RUNNING" = true ]; then
    show_status "RunPod LLM Server" "done"
else
    show_status "RunPod LLM Server" "fail"
fi

if [ "$RENDER_UPDATED" = true ]; then
    show_status "Render Backend Updated" "done"
else
    show_status "Render Backend Updated" "todo"
fi

if [ "$BACKEND_HEALTHY" = true ]; then
    show_status "Backend Health Check" "done"
else
    show_status "Backend Health Check" "todo"
fi

if [ "$FRONTEND_WORKING" = true ]; then
    show_status "Frontend Chat Working" "done"
else
    show_status "Frontend Chat Working" "todo"
fi

echo ""

if [ "$SERVER_RUNNING" = true ] && [ "$RENDER_UPDATED" = true ] && [ "$BACKEND_HEALTHY" = true ] && [ "$FRONTEND_WORKING" = true ]; then
    echo -e "${GREEN}🎉 Congratulations! Phase 1 Day 1 Complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "- Review PHASE_1_QUICK_START.md for Day 2-5"
    echo "- Test multi-language support"
    echo "- Test all use cases"
    echo ""
else
    echo -e "${YELLOW}⚠ Phase 1 Day 1 Incomplete${NC}"
    echo ""
    echo "Please complete the failed/pending steps above."
    echo "Refer to PHASE_1_CURRENT_STATUS.md for detailed instructions."
    echo ""
fi

echo -e "${BLUE}========================================${NC}"
echo ""
echo "For detailed troubleshooting, see:"
echo "- RUNPOD_TROUBLESHOOTING.md"
echo "- PHASE_1_CURRENT_STATUS.md"
echo ""
