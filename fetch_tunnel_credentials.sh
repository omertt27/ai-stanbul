#!/bin/bash

echo "📥 Fetch Tunnel Credentials from Cloudflare"
echo "==========================================="
echo ""

TUNNEL_ID="3c9f3076-300f-4a61-b923-cf7be81e2919"
ACCOUNT_ID="ae70d7d9f126ec7201b9233c43ee2441"

echo "You need a Cloudflare API Token with tunnel permissions."
echo "Get it from: https://dash.cloudflare.com/profile/api-tokens"
echo ""
echo "Required permissions:"
echo "  - Account > Cloudflare Tunnel > Read"
echo ""
read -p "Enter your Cloudflare API Token: " CF_TOKEN

if [ -z "$CF_TOKEN" ]; then
    echo "❌ No token provided"
    exit 1
fi

echo ""
echo "🔍 Fetching tunnel credentials from Cloudflare..."
echo ""

RESPONSE=$(curl -s -X GET \
  "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/cfd_tunnel/${TUNNEL_ID}/token" \
  -H "Authorization: Bearer ${CF_TOKEN}" \
  -H "Content-Type: application/json")

# Check if successful
if echo "$RESPONSE" | grep -q '"success":true'; then
    echo "✅ Successfully fetched credentials!"
    echo ""
    
    # Extract and save the token
    TOKEN_VALUE=$(echo "$RESPONSE" | jq -r '.result')
    
    # The token is base64 encoded JSON, decode and save it
    echo "$TOKEN_VALUE" | base64 -d > ~/.cloudflared/${TUNNEL_ID}.json
    
    echo "✅ Credentials saved to: ~/.cloudflared/${TUNNEL_ID}.json"
    echo ""
    echo "Content:"
    cat ~/.cloudflared/${TUNNEL_ID}.json
    echo ""
    echo ""
    echo "🚀 Now you can start the tunnel!"
else
    echo "❌ Failed to fetch credentials"
    echo ""
    echo "Response:"
    echo "$RESPONSE" | jq .
    echo ""
    echo "Possible issues:"
    echo "  1. API token doesn't have tunnel read permissions"
    echo "  2. Tunnel doesn't exist or was deleted"
    echo "  3. Wrong account ID"
fi
