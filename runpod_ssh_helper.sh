#!/bin/bash

# ============================================
# RunPod SSH Helper Script
# ============================================
# Quick access to your RunPod pod for troubleshooting

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# RunPod SSH Configuration
RUNPOD_SSH="ytc61lal7ag5sy-64410fe8@ssh.runpod.io"
RUNPOD_SSH_DIRECT="root@194.68.245.173"
RUNPOD_PORT="22001"
SSH_KEY="~/.ssh/id_ed25519"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  RunPod SSH Helper${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if SSH key exists
if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
    echo -e "${YELLOW}⚠ SSH key not found at ~/.ssh/id_ed25519${NC}"
    echo "Please ensure your SSH key is set up correctly."
    echo ""
    exit 1
fi

# Show menu
echo "Choose connection method:"
echo ""
echo "1) Standard SSH (via RunPod proxy)"
echo "2) Direct TCP SSH (for SCP/SFTP)"
echo "3) Check LLM server status (via SSH)"
echo "4) View LLM server logs (via SSH)"
echo "5) Restart LLM server (via SSH)"
echo "6) Exit"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo -e "${GREEN}Connecting via RunPod proxy...${NC}"
        echo -e "Command: ssh ${RUNPOD_SSH} -i ${SSH_KEY}"
        echo ""
        ssh ${RUNPOD_SSH} -i ${SSH_KEY}
        ;;
    2)
        echo -e "${GREEN}Connecting via direct TCP...${NC}"
        echo -e "Command: ssh ${RUNPOD_SSH_DIRECT} -p ${RUNPOD_PORT} -i ${SSH_KEY}"
        echo ""
        ssh ${RUNPOD_SSH_DIRECT} -p ${RUNPOD_PORT} -i ${SSH_KEY}
        ;;
    3)
        echo -e "${GREEN}Checking LLM server status...${NC}"
        echo ""
        ssh ${RUNPOD_SSH} -i ${SSH_KEY} << 'ENDSSH'
            echo "=== Python Processes ==="
            ps aux | grep python | grep -v grep
            echo ""
            echo "=== Local Health Check ==="
            curl -s http://localhost:8888/health || echo "Server not responding locally"
            echo ""
            echo "=== Disk Usage ==="
            df -h /workspace
            echo ""
            echo "=== GPU Status ==="
            nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || echo "nvidia-smi not available"
ENDSSH
        ;;
    4)
        echo -e "${GREEN}Viewing LLM server logs (last 50 lines)...${NC}"
        echo ""
        ssh ${RUNPOD_SSH} -i ${SSH_KEY} << 'ENDSSH'
            if [ -f /workspace/server.log ]; then
                tail -50 /workspace/server.log
            else
                echo "No server.log found in /workspace"
            fi
ENDSSH
        ;;
    5)
        echo -e "${YELLOW}This will restart your LLM server.${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            echo -e "${GREEN}Restarting LLM server...${NC}"
            ssh ${RUNPOD_SSH} -i ${SSH_KEY} << 'ENDSSH'
                echo "Stopping existing server..."
                pkill -f "python.*llm_api_server" || echo "No server running"
                sleep 2
                
                echo "Starting new server..."
                cd /workspace
                nohup python llm_api_server_4bit.py > server.log 2>&1 &
                
                echo "Waiting 10 seconds for startup..."
                sleep 10
                
                echo "Testing server..."
                curl -s http://localhost:8888/health || echo "Server not responding yet"
                
                echo ""
                echo "Check logs with: tail -f /workspace/server.log"
ENDSSH
            echo -e "${GREEN}✅ Server restart command sent${NC}"
            echo "Wait 15-20 seconds, then test with:"
            echo "curl https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/health"
        else
            echo "Restart cancelled."
        fi
        ;;
    6)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
