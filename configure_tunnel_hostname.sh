#!/bin/bash

echo "🌐 Configure Public Hostname for Tunnel"
echo "========================================"
echo ""

TUNNEL_ID="3c9f3076-300f-4a61-b923-cf7be81e2919"
ACCOUNT_ID="ae70d7d9f126ec7201b9233c43ee2441"

echo "This will create a public hostname route for your tunnel."
echo ""
read -p "Enter your Cloudflare API Token: " CF_TOKEN

if [ -z "$CF_TOKEN" ]; then
    echo "❌ No token provided"
    exit 1
fi

echo ""
echo "🔍 Creating public hostname route..."
echo ""

# Create a catch-all route for the tunnel
RESPONSE=$(curl -s -X PUT \
  "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/cfd_tunnel/${TUNNEL_ID}/configurations" \
  -H "Authorization: Bearer ${CF_TOKEN}" \
  -H "Content-Type: application/json" \
  --data '{
    "config": {
      "ingress": [
        {
          "hostname": "*",
          "service": "http://localhost:8000"
        },
        {
          "service": "http_status:404"
        }
      ]
    }
  }')

if echo "$RESPONSE" | grep -q '"success":true'; then
    echo "✅ Configuration updated!"
    echo ""
    echo "Your tunnel is now accessible at:"
    echo "  https://${TUNNEL_ID}.cfargotunnel.com/health"
    echo ""
    echo "Wait 30 seconds for DNS propagation, then test:"
    echo "  curl https://${TUNNEL_ID}.cfargotunnel.com/health"
else
    echo "❌ Failed to update configuration"
    echo ""
    echo "Response:"
    echo "$RESPONSE" | jq .
fi
