#!/bin/bash
# Test if tunnel setup completed on RunPod
# Run this in your RunPod terminal to verify the setup

echo "🔍 Verifying Tunnel Setup Status"
echo "================================="
echo ""

TUNNEL_ID="3c9f3076-300f-4a61-b923-cf7be81e2919"
TUNNEL_DOMAIN="asdweq123.org"

# Check if config was updated
echo "1️⃣ Checking config.yml..."
if grep -q "hostname: ${TUNNEL_DOMAIN}" ~/.cloudflared/config.yml 2>/dev/null; then
    echo "✅ Config has hostname configured"
    echo "Config:"
    cat ~/.cloudflared/config.yml
else
    echo "❌ Config missing hostname"
    echo "Current config:"
    cat ~/.cloudflared/config.yml
    echo ""
    echo "FIX NEEDED: Run the fix script from FIX_CLOUDFLARE_TUNNEL.md"
fi
echo ""

# Check if cloudflared is running
echo "2️⃣ Checking if cloudflared is running..."
if ps aux | grep cloudflared | grep -v grep > /dev/null; then
    echo "✅ Cloudflared is running"
    ps aux | grep cloudflared | grep -v grep | head -1
else
    echo "❌ Cloudflared is NOT running"
fi
echo ""

# Check cloudflared logs
echo "3️⃣ Checking cloudflared logs..."
if [ -f /workspace/cloudflared.log ]; then
    echo "Recent logs:"
    tail -20 /workspace/cloudflared.log
else
    echo "⚠️  No log file found at /workspace/cloudflared.log"
fi
echo ""

# Test DNS route command
echo "4️⃣ Checking DNS routing..."
echo "Running: cloudflared tunnel route dns ${TUNNEL_ID} ${TUNNEL_DOMAIN}"
cloudflared tunnel route dns ${TUNNEL_ID} ${TUNNEL_DOMAIN} 2>&1
echo ""

# Wait and test
echo "5️⃣ Testing tunnel endpoint (waiting 30 seconds for DNS)..."
sleep 30

curl -s -w "\nHTTP Code: %{http_code}\n" https://${TUNNEL_DOMAIN}/health --connect-timeout 10

echo ""
echo "================================="
echo "If HTTP Code is 200, tunnel is working! ✅"
echo "If HTTP Code is 000 or connection failed, DNS needs more time or manual setup ⚠️"
