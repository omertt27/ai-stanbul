#!/bin/bash
# Run this on RunPod to check tunnel status

echo "🔍 Checking Cloudflare Tunnel on RunPod"
echo "========================================"
echo ""

echo "1️⃣ Tunnel Process Status:"
ps aux | grep cloudflared | grep -v grep
echo ""

echo "2️⃣ Last 50 lines of tunnel logs:"
tail -50 /workspace/cloudflared.log
echo ""

echo "3️⃣ Recent connection attempts (last 20 lines):"
tail -20 /workspace/cloudflared.log | grep -E "request|error|connection|ingress"
echo ""

echo "4️⃣ Current tunnel configuration:"
cat ~/.cloudflared/config.yml
echo ""

echo "5️⃣ Test vLLM directly (should work):"
curl -s http://localhost:8000/health | head -20
echo ""

echo "========================================"
echo "📋 Next Steps:"
echo "1. If tunnel is running but logs show errors → restart tunnel"
echo "2. If vLLM test fails → restart vLLM"
echo "3. If logs show 'no ingress for api.asdweq123.org' → config issue"
echo "4. If everything looks good → check Cloudflare dashboard settings"
