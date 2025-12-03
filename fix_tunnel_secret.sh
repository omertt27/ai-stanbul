#!/bin/bash

echo "🔧 Fixing Tunnel Secret Format"
echo "==============================="
echo ""

 # The TunnelSecret needs to be base64 encoded
# Your current secret is in UUID format, we need to convert it

TUNNEL_ID="3c9f3076-300f-4a61-b923-cf7be81e2919"
ACCOUNT_TAG="ae70d7d9f126ec7201b9233c43ee2441"
TUNNEL_SECRET_UUID="2eead50e-8fda-4bc1-9e60-3117c8c71848"

echo "Current tunnel secret (UUID format): $TUNNEL_SECRET_UUID"
echo ""

# Convert UUID to base64 (remove hyphens and encode)
# UUID format: 2eead50e-8fda-4bc1-9e60-3117c8c71848
# Need to convert to base64

# Method 1: Convert hex to base64
TUNNEL_SECRET_HEX=$(echo "$TUNNEL_SECRET_UUID" | tr -d '-')
echo "Hex format: $TUNNEL_SECRET_HEX"

# Convert hex to base64
TUNNEL_SECRET_BASE64=$(echo "$TUNNEL_SECRET_HEX" | xxd -r -p | base64)
echo "Base64 format: $TUNNEL_SECRET_BASE64"
echo ""

# Create the corrected credentials file
CREDENTIALS_FILE=~/.cloudflared/$TUNNEL_ID.json

echo "Creating corrected credentials file..."
cat > "$CREDENTIALS_FILE" << EOF
{
  "AccountTag": "$ACCOUNT_TAG",
  "TunnelSecret": "$TUNNEL_SECRET_BASE64",
  "TunnelID": "$TUNNEL_ID"
}
EOF

echo "✅ Credentials file updated!"
echo ""
echo "New content:"
cat "$CREDENTIALS_FILE"
echo ""
echo "Now you can run the tunnel!"
