#!/bin/bash

echo "🔄 Restarting Cloudflare Tunnel on RunPod"
echo "==========================================="
echo ""

# Stop any existing tunnel process
echo "🛑 Stopping existing tunnel..."
pkill -f cloudflared
sleep 3

# Verify it's stopped
if ps aux | grep -v grep | grep cloudflared > /dev/null; then
    echo "⚠️  Warning: cloudflared still running, force killing..."
    pkill -9 -f cloudflared
    sleep 2
fi

echo "✅ Tunnel stopped"
echo ""

# Create logs directory if it doesn't exist
mkdir -p /workspace/logs

# Start tunnel with the tunnel ID (it will fetch the API-configured route)
echo "🚀 Starting tunnel with ID: 3c9f3076-300f-4a61-b923-cf7be81e2919"
echo "   This will use the route configured via API..."
echo ""

nohup cloudflared tunnel run 3c9f3076-300f-4a61-b923-cf7be81e2919 > /workspace/logs/cloudflare-tunnel.log 2>&1 &
TUNNEL_PID=$!

# Save PID immediately
echo $TUNNEL_PID > /workspace/cloudflare-tunnel.pid

echo "✅ Tunnel process started with PID: $TUNNEL_PID"
echo ""

# Wait a moment for initial startup
echo "⏳ Waiting 5 seconds for initial startup..."
sleep 5

# Check if process is still running (early failure detection)
if ! ps -p $TUNNEL_PID > /dev/null; then
    echo ""
    echo "❌ ERROR: Tunnel process died immediately!"
    echo ""
    echo "📋 Error logs:"
    echo "=============="
    cat /workspace/logs/cloudflare-tunnel.log
    echo ""
    echo "=============================================="
    echo "🔍 Common issues:"
    echo "   1. Tunnel credentials not found or invalid"
    echo "   2. Tunnel ID doesn't exist or was deleted"
    echo "   3. Invalid tunnel configuration"
    echo ""
    echo "💡 Solutions:"
    echo "   - Check if tunnel exists: cloudflared tunnel list"
    echo "   - Verify credentials in ~/.cloudflared/"
    echo "   - Check tunnel config: cloudflared tunnel info 3c9f3076-300f-4a61-b923-cf7be81e2919"
    echo "=============================================="
    exit 1
fi

# Continue waiting for full startup
echo "⏳ Waiting 10 more seconds for tunnel to establish connections..."
sleep 10

# Check status again
echo ""
echo "📊 Tunnel Status:"
echo "=================="
if ps -p $TUNNEL_PID > /dev/null; then
    echo "✅ Process running (PID: $TUNNEL_PID)"
else
    echo "❌ Process not running!"
    echo ""
    echo "📋 Recent log entries:"
    tail -n 30 /workspace/logs/cloudflare-tunnel.log
    exit 1
fi

echo ""
echo "📋 Recent log entries:"
tail -n 20 /workspace/logs/cloudflare-tunnel.log

echo ""
echo "=============================================="
echo "🎯 Test URLs:"
echo "=============================================="
echo ""
echo "Local test:"
echo "  curl http://localhost:8000/health"
echo ""
echo "Direct Cloudflare tunnel URL:"
echo "  curl https://3c9f3076-300f-4a61-b923-cf7be81e2919.cfargotunnel.com/health"
echo ""
echo "=============================================="
echo ""
echo "💡 To monitor logs in real-time:"
echo "   tail -f /workspace/logs/cloudflare-tunnel.log"
echo ""
