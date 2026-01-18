#!/bin/bash

echo "🚀 Start Cloudflare Tunnel (Token-Based)"
echo "========================================="
echo ""

# This method doesn't require cert.pem
# It runs the tunnel using the credentials file directly

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

# Check if credentials file exists
CREDENTIALS_FILE=~/.cloudflared/3c9f3076-300f-4a61-b923-cf7be81e2919.json

if [ ! -f "$CREDENTIALS_FILE" ]; then
    echo "❌ ERROR: Credentials file not found!"
    echo "   Expected: $CREDENTIALS_FILE"
    echo ""
    echo "Please create the credentials file with your tunnel credentials."
    exit 1
fi

echo "✅ Credentials file found"
echo ""

# Start tunnel using --credentials-file flag (bypasses need for cert.pem)
echo "🚀 Starting tunnel with credentials file..."
echo "   Tunnel ID: 3c9f3076-300f-4a61-b923-cf7be81e2919"
echo "   Service: http://localhost:8000"
echo ""

nohup cloudflared tunnel --credentials-file "$CREDENTIALS_FILE" run \
    --url http://localhost:8000 \
    3c9f3076-300f-4a61-b923-cf7be81e2919 \
    > /workspace/logs/cloudflare-tunnel.log 2>&1 &

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
    echo "🔍 Troubleshooting:"
    echo "   1. Check if localhost:8000 is running"
    echo "   2. Verify credentials file is valid"
    echo "   3. Check tunnel exists in Cloudflare dashboard"
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
echo "🎯 Test Commands:"
echo "=============================================="
echo ""
echo "Local test:"
echo "  curl http://localhost:8000/health"
echo ""
echo "Check tunnel logs:"
echo "  tail -f /workspace/logs/cloudflare-tunnel.log"
echo ""
echo "Check tunnel status in Cloudflare:"
echo "  https://one.dash.cloudflare.com/"
echo ""
echo "=============================================="
echo ""
