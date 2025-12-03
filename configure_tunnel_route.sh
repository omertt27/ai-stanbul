#!/bin/bash

echo "🔧 Configuring Cloudflare Tunnel Route via API"
echo "==============================================="
echo ""
echo "This will configure your tunnel to enable the direct .cfargotunnel.com URL"
echo ""

# Get API token from user
read -p "Enter your Cloudflare API Token (from dash.cloudflare.com/profile/api-tokens): " API_TOKEN

if [ -z "$API_TOKEN" ]; then
    echo "❌ API token is required!"
    echo ""
    echo "Get your token from: https://dash.cloudflare.com/profile/api-tokens"
    echo "Create a token with 'Cloudflare Zero Trust' permissions"
    exit 1
fi

ACCOUNT_ID="ae70d7d9f126ec7201b9233c43ee2441"
TUNNEL_ID="3c9f3076-300f-4a61-b923-cf7be81e2919"

echo ""
echo "📡 Configuring tunnel route..."
echo "Account ID: $ACCOUNT_ID"
echo "Tunnel ID: $TUNNEL_ID"
echo ""

# Configure ingress rule via API
RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X PUT \
  "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/cfd_tunnel/${TUNNEL_ID}/configurations" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Content-Type: application/json" \
  --data '{
    "config": {
      "ingress": [
        {
          "service": "http://localhost:8000"
        }
      ]
    }
  }')

# Extract HTTP status
HTTP_STATUS=$(echo "$RESPONSE" | grep "HTTP_STATUS:" | cut -d: -f2)
BODY=$(echo "$RESPONSE" | sed '/HTTP_STATUS:/d')

echo "📋 API Response (HTTP $HTTP_STATUS):"
echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
echo ""

# Check if successful
if echo "$BODY" | grep -q '"success":true'; then
    echo "✅ Route configured successfully!"
    echo ""
    echo "⏳ Waiting 30 seconds for propagation..."
    sleep 30
    
    echo ""
    echo "🧪 Testing direct URL..."
    HEALTH_RESPONSE=$(curl -s -w "\nHTTP_CODE:%{http_code}" https://3c9f3076-300f-4a61-b923-cf7be81e2919.cfargotunnel.com/health)
    HEALTH_CODE=$(echo "$HEALTH_RESPONSE" | grep "HTTP_CODE:" | cut -d: -f2)
    HEALTH_BODY=$(echo "$HEALTH_RESPONSE" | sed '/HTTP_CODE:/d')
    
    if [ "$HEALTH_CODE" = "200" ]; then
        echo "✅ SUCCESS! Tunnel URL is working!"
        echo ""
        echo "Response: $HEALTH_BODY"
    else
        echo "⚠️  URL returned HTTP $HEALTH_CODE"
        echo "Response: $HEALTH_BODY"
        echo ""
        echo "💡 The route is configured, but may need more time to propagate."
        echo "Try again in 1-2 minutes:"
        echo "curl https://3c9f3076-300f-4a61-b923-cf7be81e2919.cfargotunnel.com/health"
    fi
    
    echo ""
    echo "================================================"
    echo "🎉 Your production-ready LLM URL:"
    echo "https://3c9f3076-300f-4a61-b923-cf7be81e2919.cfargotunnel.com"
    echo "================================================"
    echo ""
    echo "📝 Update your environment variables:"
    echo ""
    echo "Render Backend:"
    echo "LLM_SERVER_URL=https://3c9f3076-300f-4a61-b923-cf7be81e2919.cfargotunnel.com"
    echo ""
    echo "Vercel Frontend:"
    echo "VITE_LLM_SERVER_URL=https://3c9f3076-300f-4a61-b923-cf7be81e2919.cfargotunnel.com"
    echo ""
else
    echo "❌ Configuration failed!"
    echo ""
    
    if echo "$BODY" | grep -qi "authentication"; then
        echo "Error: Invalid API token or insufficient permissions"
        echo ""
        echo "Solution:"
        echo "1. Go to: https://dash.cloudflare.com/profile/api-tokens"
        echo "2. Create token with 'Cloudflare Zero Trust' Edit permissions"
        echo "3. Run this script again with the new token"
    elif echo "$BODY" | grep -qi "not found"; then
        echo "Error: Tunnel not found"
        echo ""
        echo "Solution: Make sure the tunnel is running on RunPod"
    else
        echo "Check the API response above for details"
    fi
    
    echo ""
    echo "================================================"
    echo "💡 Alternative: Use RunPod URL for now"
    echo "================================================"
    echo ""
    echo "https://4r1su4zfuok0s7-19123.proxy.runpod.net/nsvmrgaqgp8j29z4dytki4habch6tazn"
    echo ""
    echo "This works immediately while we debug the Cloudflare route."
fi
