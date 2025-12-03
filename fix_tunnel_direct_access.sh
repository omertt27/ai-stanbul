#!/bin/bash

echo "🔧 Fixing Cloudflare Tunnel for Direct Access"
echo "=============================================="
echo ""

# Check if config exists
if [ ! -f ~/.cloudflared/config.yml ]; then
    echo "❌ Config file not found at ~/.cloudflared/config.yml"
    exit 1
fi

echo "📋 Current tunnel configuration:"
cat ~/.cloudflared/config.yml
echo ""
echo "=============================================="
echo ""

# Backup existing config
echo "💾 Backing up current config..."
cp ~/.cloudflared/config.yml ~/.cloudflared/config.yml.backup
echo "✅ Backup saved to ~/.cloudflared/config.yml.backup"
echo ""

# Create new config with catch-all ingress
echo "✏️  Creating new configuration with catch-all rule..."
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61
credentials-file: /root/.cloudflared/5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.json

ingress:
  - hostname: llm.aistanbul.net
    service: http://localhost:8000
  - service: http://localhost:8000
    # This is the catch-all rule - enables direct .cfargotunnel.com access

# Optional: Enable better logging
# loglevel: debug
EOF

echo "✅ New configuration created!"
echo ""

echo "📋 New tunnel configuration:"
cat ~/.cloudflared/config.yml
echo ""
echo "=============================================="
echo ""

# Stop existing tunnel
echo "🛑 Stopping existing tunnel..."
if [ -f /workspace/cloudflare-tunnel.pid ]; then
    OLD_PID=$(cat /workspace/cloudflare-tunnel.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        kill $OLD_PID
        echo "✅ Stopped old tunnel (PID: $OLD_PID)"
        sleep 2
    else
        echo "ℹ️  No running tunnel found with PID: $OLD_PID"
    fi
else
    echo "ℹ️  No PID file found, checking for running processes..."
    pkill -f cloudflared
    sleep 2
fi

# Start tunnel with new config
echo ""
echo "🚀 Starting tunnel with new configuration..."
nohup cloudflared tunnel --config ~/.cloudflared/config.yml run 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61 > /workspace/logs/cloudflare-tunnel.log 2>&1 &
NEW_PID=$!
echo $NEW_PID > /workspace/cloudflare-tunnel.pid
echo "✅ Tunnel started with PID: $NEW_PID"
echo ""

# Wait for tunnel to initialize
echo "⏳ Waiting for tunnel to initialize (10 seconds)..."
sleep 10

# Check if tunnel is running
if ps -p $NEW_PID > /dev/null 2>&1; then
    echo "✅ Tunnel process is running!"
else
    echo "❌ Tunnel process is not running!"
    echo ""
    echo "📋 Recent logs:"
    tail -20 /workspace/logs/cloudflare-tunnel.log
    exit 1
fi

# Check tunnel logs for connections
echo ""
echo "📋 Recent tunnel activity:"
tail -10 /workspace/logs/cloudflare-tunnel.log | grep -E "Registered|connection"
echo ""

# Test local server
echo "🧪 Testing local LLM server..."
LOCAL_TEST=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$LOCAL_TEST" = "200" ]; then
    echo "✅ Local server responding (HTTP $LOCAL_TEST)"
else
    echo "⚠️  Local server returned HTTP $LOCAL_TEST"
    echo "   Make sure llm_server.py is running!"
fi

echo ""
echo "=============================================="
echo "✅ Tunnel configuration updated!"
echo "=============================================="
echo ""
echo "🧪 Test your direct tunnel URL (wait 30 seconds):"
echo ""
echo "curl https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com/health"
echo ""
echo "📋 Useful commands:"
echo ""
echo "# Check tunnel status:"
echo "ps aux | grep cloudflared"
echo ""
echo "# View live tunnel logs:"
echo "tail -f /workspace/logs/cloudflare-tunnel.log"
echo ""
echo "# Check for errors:"
echo "grep -i error /workspace/logs/cloudflare-tunnel.log"
echo ""
echo "# Restore backup if needed:"
echo "cp ~/.cloudflared/config.yml.backup ~/.cloudflared/config.yml"
echo ""
