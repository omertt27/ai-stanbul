#!/bin/bash
# Run on RunPod to check and fix hostname configuration

echo "🔍 Checking Hostname Configuration"
echo "===================================="
echo ""

# Check what Cloudflare thinks the hostname is
echo "1️⃣ Current config.yml on disk:"
cat ~/.cloudflared/config.yml
echo ""

echo "2️⃣ What cloudflared is actually using (from logs):"
grep -i "hostname" /workspace/cloudflared.log | tail -5
echo ""

echo "3️⃣ The hostname mismatch:"
echo "   Config file says: asdweq123.org"
echo "   Cloudflare is using: api.asdweq123.org"
echo ""

echo "4️⃣ We have TWO options:"
echo ""
echo "Option A: Use api.asdweq123.org (already working in Cloudflare)"
echo "   - No changes needed on RunPod"
echo "   - Update .env: LLM_API_URL=https://api.asdweq123.org"
echo ""
echo "Option B: Change to asdweq123.org (root domain)"
echo "   - Need to update Cloudflare DNS"
echo "   - Restart tunnel"
echo ""

echo "5️⃣ Testing api.asdweq123.org from RunPod:"
curl -s -w "\nHTTP Code: %{http_code}\n" http://localhost:8000/health
echo ""

echo "===================================="
echo "Recommendation: Use api.asdweq123.org ✅"
echo "It's already configured in Cloudflare DNS"
