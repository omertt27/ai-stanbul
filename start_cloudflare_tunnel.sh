#!/bin/bash

# 🚀 Start Cloudflare Tunnel with Token (RunPod)
# This script starts the Cloudflare tunnel using nohup since RunPod doesn't use systemd

echo "🚀 Starting Cloudflare Tunnel..."

# Replace YOUR_TOKEN with your actual token from Cloudflare dashboard
TOKEN="YOUR_TOKEN_HERE"

# Check if token is still placeholder
if [ "$TOKEN" == "YOUR_TOKEN_HERE" ]; then
    echo "❌ ERROR: Please edit this script and replace YOUR_TOKEN_HERE with your actual token!"
    echo ""
    echo "Get your token from:"
    echo "1. Go to Cloudflare Dashboard"
    echo "2. Zero Trust → Networks → Tunnels"
    echo "3. Click your tunnel"
    echo "4. Click 'Configure'"
    echo "5. Copy the token from the install command"
    exit 1
fi

# Create logs directory
mkdir -p /workspace/logs

# Check if tunnel is already running
if [ -f /workspace/cloudflare-tunnel.pid ]; then
    OLD_PID=$(cat /workspace/cloudflare-tunnel.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "⚠️  Tunnel is already running (PID: $OLD_PID)"
        echo "Stop it first with: kill $OLD_PID"
        exit 1
    fi
fi

# Start tunnel with nohup
echo "Starting tunnel with token..."
nohup cloudflared tunnel --no-autoupdate run --token $TOKEN \
  > /workspace/logs/cloudflare-tunnel.log 2>&1 &

# Save PID
TUNNEL_PID=$!
echo $TUNNEL_PID > /workspace/cloudflare-tunnel.pid

# Disown so it keeps running
disown

echo "✅ Tunnel started successfully!"
echo ""
echo "PID: $TUNNEL_PID"
echo "Log file: /workspace/logs/cloudflare-tunnel.log"
echo ""
echo "📋 Useful commands:"
echo "  View logs:  tail -f /workspace/logs/cloudflare-tunnel.log"
echo "  Check:      ps aux | grep cloudflared"
echo "  Stop:       kill $TUNNEL_PID"
echo ""
echo "🌐 Next steps:"
echo "1. Configure public hostname in Cloudflare dashboard"
echo "2. Test: curl https://llm.yourdomain.com/health"
