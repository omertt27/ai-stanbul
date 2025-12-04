#!/bin/bash
# Run this on RunPod terminal to diagnose tunnel issue

echo "🔍 Diagnosing Tunnel Connection Issue"
echo "======================================"
echo ""

# 1. Check if vLLM is responding locally
echo "1️⃣ Testing vLLM locally:"
curl -s http://localhost:8000/health
echo ""

# 2. Check cloudflared process
echo "2️⃣ Cloudflared process:"
ps aux | grep cloudflared | grep -v grep
echo ""

# 3. Check recent cloudflared logs
echo "3️⃣ Recent cloudflared logs (last 30 lines):"
tail -30 /workspace/cloudflared.log
echo ""

# 4. Check config
echo "4️⃣ Current config:"
cat ~/.cloudflared/config.yml
echo ""

# 5. Check if tunnel is getting requests
echo "5️⃣ Checking for incoming requests in logs:"
grep -i "request" /workspace/cloudflared.log | tail -10 || echo "No requests found in logs"
echo ""

# 6. Check for errors
echo "6️⃣ Checking for errors in logs:"
grep -i "error\|fail\|refused" /workspace/cloudflared.log | tail -10 || echo "No errors found"
echo ""

echo "======================================"
echo "If you see errors above, the tunnel needs to be restarted."
echo "If vLLM is healthy but tunnel has errors, restart with:"
echo "  pkill cloudflared"
echo "  sleep 2"
echo "  nohup cloudflared tunnel run 3c9f3076-300f-4a61-b923-cf7be81e2919 > /workspace/cloudflared.log 2>&1 &"
