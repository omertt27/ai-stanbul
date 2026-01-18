#!/bin/bash

echo "📤 Copy Cloudflare Certificate to RunPod"
echo "========================================="
echo ""

# Check if local cert exists
if [ ! -f ~/.cloudflared/cert.pem ]; then
    echo "❌ ERROR: Local cert.pem not found in ~/.cloudflared/"
    echo ""
    echo "You need to get the certificate first:"
    echo "  1. Run: cloudflared tunnel login"
    echo "  2. This will create ~/.cloudflared/cert.pem"
    echo "  3. Then run this script again"
    echo ""
    exit 1
fi

echo "✅ Local certificate found"
echo ""

# Get RunPod SSH details
read -p "Enter RunPod SSH port (e.g., 12345): " RUNPOD_PORT
read -p "Enter RunPod public IP (e.g., 123.45.67.89): " RUNPOD_IP

echo ""
echo "📋 Transfer details:"
echo "   Source: ~/.cloudflared/cert.pem"
echo "   Target: root@${RUNPOD_IP}:${RUNPOD_PORT}:~/.cloudflared/"
echo ""
read -p "Proceed with transfer? (y/n): " CONFIRM

if [ "$CONFIRM" != "y" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🚀 Copying certificate to RunPod..."
echo ""

# Copy the certificate
scp -P "$RUNPOD_PORT" ~/.cloudflared/cert.pem "root@${RUNPOD_IP}:~/.cloudflared/"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Certificate copied successfully!"
    echo ""
    echo "Next steps on RunPod:"
    echo "  1. Verify: ls -la ~/.cloudflared/"
    echo "  2. Run: ./restart_tunnel_on_runpod.sh"
    echo ""
else
    echo ""
    echo "❌ Transfer failed!"
    echo ""
    echo "Troubleshooting:"
    echo "  - Check SSH connection: ssh -p $RUNPOD_PORT root@$RUNPOD_IP"
    echo "  - Verify RunPod SSH is enabled"
    echo "  - Check port and IP are correct"
    echo ""
fi
