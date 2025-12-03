#!/bin/bash
# Quick Connect to RunPod Instance
# Save this file as: connect_runpod.sh
# Usage: ./connect_runpod.sh

echo "🚀 RunPod Quick Connection Script"
echo "=================================="
echo ""

# Connection details
PROXY_HOST="4r1su4zfuok0s7-64410d62@ssh.runpod.io"
DIRECT_HOST="root@194.68.245.153"
DIRECT_PORT="22077"
SSH_KEY="$HOME/.ssh/id_ed25519"
WEB_TERMINAL_PORT="19123"

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found at: $SSH_KEY"
    echo ""
    echo "Please generate an SSH key first:"
    echo "  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519"
    echo ""
    exit 1
fi

# Show menu
echo "Select connection method:"
echo ""
echo "1) SSH via RunPod Proxy (Recommended)"
echo "2) Direct TCP Connection (Faster, for file transfers)"
echo "3) Open Web Terminal in Browser"
echo "4) Copy SSH Commands"
echo "5) Exit"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "🔌 Connecting via RunPod Proxy..."
        echo "Command: ssh $PROXY_HOST -i $SSH_KEY"
        echo ""
        ssh $PROXY_HOST -i $SSH_KEY
        ;;
    2)
        echo ""
        echo "🔌 Connecting via Direct TCP..."
        echo "Command: ssh $DIRECT_HOST -p $DIRECT_PORT -i $SSH_KEY"
        echo ""
        ssh $DIRECT_HOST -p $DIRECT_PORT -i $SSH_KEY
        ;;
    3)
        echo ""
        echo "🌐 Web Terminal Access..."
        echo "Port: $WEB_TERMINAL_PORT"
        echo ""
        echo "Access via RunPod Dashboard:"
        echo "  1. Go to your RunPod dashboard"
        echo "  2. Find your pod"
        echo "  3. Click 'Open Web Terminal' button"
        echo ""
        ;;
    4)
        echo ""
        echo "📋 SSH Commands:"
        echo ""
        echo "# Proxy Connection (SSH only, no SCP/SFTP):"
        echo "ssh $PROXY_HOST -i $SSH_KEY"
        echo ""
        echo "# Direct Connection (Supports SCP/SFTP):"
        echo "ssh $DIRECT_HOST -p $DIRECT_PORT -i $SSH_KEY"
        echo ""
        echo "# Web Terminal Port: $WEB_TERMINAL_PORT"
        echo "# Access via RunPod Dashboard → Your Pod → 'Open Web Terminal'"
        echo ""
        echo "# File Upload (SCP via Direct Connection):"
        echo "scp -P $DIRECT_PORT -i $SSH_KEY local_file.txt $DIRECT_HOST:/workspace/"
        echo ""
        echo "# File Download (SCP via Direct Connection):"
        echo "scp -P $DIRECT_PORT -i $SSH_KEY $DIRECT_HOST:/workspace/file.txt ./"
        echo ""
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✅ Connection closed"
