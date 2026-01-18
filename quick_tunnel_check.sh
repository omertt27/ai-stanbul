#!/bin/bash
# Quick Tunnel Check - Run these commands in RunPod terminal

echo "=== Quick Tunnel Status Check ==="
echo ""

# 1. Check vLLM
echo "1. vLLM Process:"
ps aux | grep vllm | grep -v grep || echo "   ❌ vLLM not running"
echo ""

# 2. Check port 8000
echo "2. Port 8000:"
ss -tuln | grep ":8000" || netstat -tuln | grep ":8000" || echo "   ❌ Port 8000 not listening"
echo ""

# 3. Test vLLM locally
echo "3. vLLM Health Check (localhost):"
curl -s http://localhost:8000/health || echo "   ❌ vLLM not responding"
echo ""

# 4. Check tunnels
echo "4. Active Tunnels:"
echo "   Cloudflare:"
ps aux | grep cloudflared | grep -v grep || echo "      No cloudflared process"
echo "   SSH Tunnel:"
ps aux | grep "ssh.*-R.*8000" | grep -v grep || echo "      No SSH tunnel"
echo "   ngrok:"
ps aux | grep ngrok | grep -v grep || echo "      No ngrok process"
echo ""

# 5. Test external domain
echo "5. Testing asdweq123.org:"
curl -s -w "\nHTTP Code: %{http_code}\n" https://asdweq123.org/health --connect-timeout 5 || echo "   ❌ Cannot reach asdweq123.org"
echo ""

echo "=== End of Check ==="
