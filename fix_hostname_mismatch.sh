#!/bin/bash
# Fix Cloudflare Tunnel Hostname - Run on RunPod

echo "🔧 Fixing Cloudflare Tunnel Hostname Mismatch"
echo "=============================================="
echo ""

TUNNEL_ID="3c9f3076-300f-4a61-b923-cf7be81e2919"

# 1. Check current config file
echo "1️⃣ Current config.yml on disk:"
cat ~/.cloudflared/config.yml
echo ""

# 2. The issue: Cloudflare is overriding with "api.asdweq123.org"
echo "2️⃣ The Problem:"
echo "   - Your config file says: asdweq123.org"
echo "   - Cloudflare dashboard says: api.asdweq123.org"
echo "   - Cloudflare's setting wins!"
echo ""

# 3. Solution: Update Cloudflare tunnel route
echo "3️⃣ We have 2 options:"
echo ""
echo "Option A: Use Cloudflare Dashboard (EASIEST)"
echo "   1. Go to: https://one.dash.cloudflare.com"
echo "   2. Access → Tunnels"
echo "   3. Find tunnel: 3c9f3076-300f-4a61-b923-cf7be81e2919"
echo "   4. Configure → Public Hostname"
echo "   5. Change 'api.asdweq123.org' to 'asdweq123.org'"
echo "   6. Save"
echo ""
echo "Option B: Match DNS to existing tunnel (FASTER)"
echo "   - Keep tunnel as: api.asdweq123.org"
echo "   - Use: https://api.asdweq123.org in your .env"
echo ""

# 4. Check what DNS you added
echo "4️⃣ Which DNS record did you add in Cloudflare?"
echo "   If you added: asdweq123.org (root) → need Option A"
echo "   If you added: api.asdweq123.org → need Option B"
echo ""

echo "=============================================="
echo "Waiting for your decision..."
echo ""
echo "Quick test - try the api subdomain:"
echo "Run this on your Mac:"
echo "  curl -s https://api.asdweq123.org/health"
