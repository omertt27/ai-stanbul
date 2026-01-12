#!/bin/bash

# 🚀 Master Startup Script - LLM Server + Cloudflare Tunnel
# Starts both services needed for external LLM access

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 STARTING LLM SERVER + CLOUDFLARE TUNNEL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create logs directory
mkdir -p /workspace/logs

# ═══════════════════════════════════════════════
# STEP 1: Start LLM Server
# ═══════════════════════════════════════════════

echo "📦 Step 1: Starting LLM Server..."
echo "─────────────────────────────────────────────"

# Check if LLM server script exists
if [ ! -f /workspace/llm_server.py ]; then
    echo "❌ ERROR: llm_server.py not found in /workspace/"
    echo "   Make sure you uploaded the LLM server files."
    exit 1
fi

# Check if already running
if [ -f /workspace/llm_server.pid ]; then
    OLD_PID=$(cat /workspace/llm_server.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "✅ LLM Server already running (PID: $OLD_PID)"
    else
        echo "ℹ️  Removing stale PID file..."
        rm /workspace/llm_server.pid
        
        # Start LLM server
        cd /workspace
        nohup python3 llm_server.py > /workspace/logs/llm_server.log 2>&1 &
        LLM_PID=$!
        echo $LLM_PID > /workspace/llm_server.pid
        disown
        echo "✅ LLM Server started (PID: $LLM_PID)"
    fi
else
    # Start LLM server
    cd /workspace
    nohup python3 llm_server.py > /workspace/logs/llm_server.log 2>&1 &
    LLM_PID=$!
    echo $LLM_PID > /workspace/llm_server.pid
    disown
    echo "✅ LLM Server started (PID: $LLM_PID)"
fi

echo ""
echo "Waiting for LLM server to initialize..."
sleep 5

# Test if server is responding
echo "Testing LLM server..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ LLM Server is responding on port 8000"
else
    echo "⚠️  LLM Server may still be loading (this is normal for first start)"
    echo "   Check logs: tail -f /workspace/logs/llm_server.log"
fi

echo ""

# ═══════════════════════════════════════════════
# STEP 2: Start Cloudflare Tunnel
# ═══════════════════════════════════════════════

echo "🌐 Step 2: Starting Cloudflare Tunnel..."
echo "─────────────────────────────────────────────"

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

if [ "$TOKEN" == "YOUR_TOKEN_HERE" ]; then
    echo "⚠️  CLOUDFLARE TUNNEL TOKEN NOT CONFIGURED!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 To complete setup:"
    echo ""
    echo "1. Get your token from Cloudflare Dashboard:"
    echo "   • Go to: https://one.dash.cloudflare.com/"
    echo "   • Navigate: Zero Trust → Networks → Tunnels"
    echo "   • Click on tunnel: LLM"
    echo "   • Click: Configure"
    echo "   • Copy the token from install command"
    echo ""
    echo "2. Edit this script:"
    echo "   nano /workspace/start_all_services.sh"
    echo ""
    echo "3. Replace YOUR_TOKEN_HERE with your actual token"
    echo ""
    echo "4. Run this script again"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Or start tunnel manually with:"
    echo "  cd /workspace && mkdir -p logs && \\"
    echo "  nohup cloudflared tunnel --no-autoupdate run --token YOUR_TOKEN \\"
    echo "    > /workspace/logs/cloudflare-tunnel.log 2>&1 & \\"
    echo "  echo \$! > /workspace/cloudflare-tunnel.pid && disown"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "✅ LLM Server is running - you can use it locally:"
    echo "   curl http://localhost:8000/health"
    exit 1
fi

# Check if already running
if [ -f /workspace/cloudflare-tunnel.pid ]; then
    OLD_PID=$(cat /workspace/cloudflare-tunnel.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "✅ Cloudflare Tunnel already running (PID: $OLD_PID)"
    else
        echo "ℹ️  Removing stale PID file..."
        rm /workspace/cloudflare-tunnel.pid
        
        # Start tunnel
        nohup cloudflared tunnel --no-autoupdate run --token $TOKEN \
          > /workspace/logs/cloudflare-tunnel.log 2>&1 &
        TUNNEL_PID=$!
        echo $TUNNEL_PID > /workspace/cloudflare-tunnel.pid
        disown
        echo "✅ Cloudflare Tunnel started (PID: $TUNNEL_PID)"
    fi
else
    # Start tunnel
    nohup cloudflared tunnel --no-autoupdate run --token $TOKEN \
      > /workspace/logs/cloudflare-tunnel.log 2>&1 &
    TUNNEL_PID=$!
    echo $TUNNEL_PID > /workspace/cloudflare-tunnel.pid
    disown
    echo "✅ Cloudflare Tunnel started (PID: $TUNNEL_PID)"
fi

sleep 2

echo ""

# ═══════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL SERVICES STARTED!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Status:"
echo "  LLM Server PID:       $(cat /workspace/llm_server.pid 2>/dev/null || echo 'N/A')"
echo "  Cloudflare Tunnel PID: $(cat /workspace/cloudflare-tunnel.pid 2>/dev/null || echo 'N/A')"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Management Commands:"
echo ""
echo "  Check Status:"
echo "    ps aux | grep -E 'llm_server|cloudflared'"
echo ""
echo "  View Logs:"
echo "    tail -f /workspace/logs/llm_server.log"
echo "    tail -f /workspace/logs/cloudflare-tunnel.log"
echo ""
echo "  Stop Services:"
echo "    kill \$(cat /workspace/llm_server.pid)"
echo "    kill \$(cat /workspace/cloudflare-tunnel.pid)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Next Steps:"
echo ""
echo "  1. Configure public hostname in Cloudflare dashboard:"
echo "     • Go to: Zero Trust → Networks → Tunnels → LLM"
echo "     • Click: Public Hostname tab"
echo "     • Add hostname: llm.yourdomain.com → localhost:8000"
echo ""
echo "  2. Test external access:"
echo "     curl https://llm.yourdomain.com/health"
echo ""
echo "  3. Update backend .env:"
echo "     LLM_SERVER_URL=https://llm.yourdomain.com"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
