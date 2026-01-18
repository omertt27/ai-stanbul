#!/bin/bash

echo "🚀 Quick Fix: Enable Direct Tunnel Access"
echo "=========================================="
echo ""

# Create config with catch-all rule
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61
credentials-file: /root/.cloudflared/5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.json

ingress:
  - service: http://localhost:8000
EOF

echo "✅ Config updated with catch-all rule"

# Restart tunnel
kill $(cat /workspace/cloudflare-tunnel.pid) 2>/dev/null
sleep 2

nohup cloudflared tunnel --config ~/.cloudflared/config.yml run 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61 > /workspace/logs/cloudflare-tunnel.log 2>&1 &
echo $! > /workspace/cloudflare-tunnel.pid

echo "✅ Tunnel restarted"
echo ""
echo "⏳ Waiting 30 seconds for tunnel to register..."
sleep 30

echo ""
echo "🧪 Testing direct URL..."
curl https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com/health

echo ""
echo ""
echo "=========================================="
echo "✅ Your Direct Tunnel URL:"
echo "https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com"
echo "=========================================="
