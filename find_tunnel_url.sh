#!/bin/bash

echo "🔍 Finding Your Cloudflare Tunnel ID..."
echo "========================================"
echo ""

# Method 1: Check config file
echo "Method 1: Checking config file..."
if [ -f ~/.cloudflared/config.yml ]; then
    TUNNEL_ID=$(grep "tunnel:" ~/.cloudflared/config.yml | awk '{print $2}')
    if [ ! -z "$TUNNEL_ID" ]; then
        echo "✅ Found in config: $TUNNEL_ID"
    fi
fi

# Method 2: Check credentials files
echo ""
echo "Method 2: Checking credentials files..."
if [ -d ~/.cloudflared ]; then
    for file in ~/.cloudflared/*.json; do
        if [ -f "$file" ]; then
            FILENAME=$(basename "$file" .json)
            echo "✅ Found credentials file: $FILENAME"
            TUNNEL_ID="$FILENAME"
        fi
    done
fi

# Method 3: Check tunnel logs
echo ""
echo "Method 3: Checking tunnel logs..."
if [ -f /workspace/logs/cloudflare-tunnel.log ]; then
    LOG_TUNNEL=$(grep "Registered tunnel" /workspace/logs/cloudflare-tunnel.log | head -1 | grep -oP 'connection=[a-f0-9\-]+' | cut -d= -f2)
    if [ ! -z "$LOG_TUNNEL" ]; then
        echo "✅ Found in logs: $LOG_TUNNEL"
    fi
fi

# Display results
echo ""
echo "========================================"
echo "🎯 YOUR TUNNEL INFORMATION"
echo "========================================"

if [ ! -z "$TUNNEL_ID" ]; then
    echo ""
    echo "Tunnel ID: $TUNNEL_ID"
    echo ""
    echo "📌 Your Direct Tunnel URL:"
    echo "https://${TUNNEL_ID}.cfargotunnel.com"
    echo ""
    echo "🧪 Test Commands:"
    echo ""
    echo "# Test health endpoint:"
    echo "curl https://${TUNNEL_ID}.cfargotunnel.com/health"
    echo ""
    echo "# Test generation:"
    echo "curl -X POST https://${TUNNEL_ID}.cfargotunnel.com/generate \\"
    echo "  -H \"Content-Type: application/json\" \\"
    echo "  -d '{\"prompt\":\"Hello\",\"max_tokens\":10}'"
    echo ""
    echo "========================================"
    echo "✅ Copy the URL above to use in your:"
    echo "   - Render backend (LLM_SERVER_URL)"
    echo "   - Vercel frontend (VITE_LLM_SERVER_URL)"
    echo "========================================"
else
    echo ""
    echo "❌ Could not automatically find tunnel ID"
    echo ""
    echo "📋 Manual steps:"
    echo "1. Go to: https://one.dash.cloudflare.com/"
    echo "2. Click: Networks → Tunnels"
    echo "3. Copy your tunnel ID (long UUID)"
    echo ""
    echo "Then your URL will be:"
    echo "https://<TUNNEL_ID>.cfargotunnel.com"
    echo ""
fi

echo ""
echo "📝 For full setup guide, see: DIRECT_TUNNEL_URL_SETUP.md"
echo ""
