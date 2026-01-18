#!/bin/bash

# 🚀 Start Cloudflare Tunnel "LLM" on RunPod
# 
# Tunnel Details:
# - Name: LLM
# - Tunnel ID: 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61
# - Connector ID: d0d73789-a679-4b70-a0d0-59b043230562

echo "🚀 Starting Cloudflare Tunnel: LLM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "❌ ERROR: cloudflared is not installed!"
    echo ""
    echo "Install it first:"
    echo "  cd /workspace"
    echo "  wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    echo "  chmod +x cloudflared-linux-amd64"
    echo "  mv cloudflared-linux-amd64 /usr/local/bin/cloudflared"
    exit 1
fi

# Token for Cloudflare Tunnel "LLM"
TOKEN="eyJhIjoiYWU3MGQ3ZDlmMTI2ZWM3MjAxYjkyMzNjNDNlZTI0NDEiLCJ0IjoiNTg4NzgwM2UtY2Y3Mi00ZmNjLTgyY2UtNGNjMWY0YjFkZDYxIiwicyI6Ik1EWmlOalZtWW1RdFpHUTVOaTAwTmpFNUxXRmlZMk10WW1FNU1HUTBOR1ZrWm1ZeSJ9"

# Check if token is still placeholder
if [ "$TOKEN" == "YOUR_TOKEN_HERE" ]; then
    echo "⚠️  You need to add your Cloudflare tunnel token to this script."
    echo ""
    echo "📋 How to get your token:"
    echo "1. Go to: https://one.dash.cloudflare.com/"
    echo "2. Navigate: Zero Trust → Networks → Tunnels"
    echo "3. Click on tunnel: LLM"
    echo "4. Click: Configure"
    echo "5. Copy the token from the install command"
    echo ""
    echo "Then edit this script and replace YOUR_TOKEN_HERE with your actual token."
    echo ""
    echo "Or run manually:"
    echo "  cd /workspace && mkdir -p logs && \\"
    echo "  nohup cloudflared tunnel --no-autoupdate run --token YOUR_TOKEN \\"
    echo "    > /workspace/logs/cloudflare-tunnel.log 2>&1 & \\"
    echo "  echo \$! > /workspace/cloudflare-tunnel.pid && disown"
    exit 1
fi

# Create logs directory
mkdir -p /workspace/logs

# Check if tunnel is already running
if [ -f /workspace/cloudflare-tunnel.pid ]; then
    OLD_PID=$(cat /workspace/cloudflare-tunnel.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "⚠️  Tunnel is already running!"
        echo "    PID: $OLD_PID"
        echo ""
        echo "To restart:"
        echo "  1. Stop: kill $OLD_PID"
        echo "  2. Run this script again"
        exit 1
    else
        echo "ℹ️  Removing stale PID file..."
        rm /workspace/cloudflare-tunnel.pid
    fi
fi

# Start tunnel with nohup
echo "Starting tunnel connector..."
nohup cloudflared tunnel --no-autoupdate run --token $TOKEN \
  > /workspace/logs/cloudflare-tunnel.log 2>&1 &

# Save PID
TUNNEL_PID=$!
echo $TUNNEL_PID > /workspace/cloudflare-tunnel.pid

# Disown so it keeps running
disown

# Wait a moment for startup
sleep 2

# Check if process is still running
if ps -p $TUNNEL_PID > /dev/null 2>&1; then
    echo "✅ Tunnel started successfully!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Status:"
    echo "  Tunnel: LLM"
    echo "  PID: $TUNNEL_PID"
    echo "  Log: /workspace/logs/cloudflare-tunnel.log"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 Useful Commands:"
    echo "  View logs:    tail -f /workspace/logs/cloudflare-tunnel.log"
    echo "  Check status: ps aux | grep cloudflared"
    echo "  Stop tunnel:  kill $TUNNEL_PID"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🌐 Next Steps:"
    echo "  1. Configure public hostname in Cloudflare dashboard"
    echo "  2. Go to: Zero Trust → Networks → Tunnels → LLM"
    echo "  3. Add public hostname: llm.yourdomain.com → localhost:8000"
    echo "  4. Test: curl https://llm.yourdomain.com/health"
    echo ""
    
    # Show first few log lines
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📜 Recent logs:"
    tail -n 10 /workspace/logs/cloudflare-tunnel.log
else
    echo "❌ ERROR: Tunnel failed to start!"
    echo ""
    echo "Check logs for errors:"
    echo "  cat /workspace/logs/cloudflare-tunnel.log"
    exit 1
fi
