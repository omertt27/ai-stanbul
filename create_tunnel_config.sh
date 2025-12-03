#!/bin/bash

echo "🔧 Creating Cloudflare Config from Scratch"
echo "==========================================="
echo ""

# Create .cloudflared directory
echo "📁 Creating .cloudflared directory..."
mkdir -p ~/.cloudflared
echo "✅ Directory created"
echo ""

# Create config file
echo "📝 Creating config.yml..."
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61
credentials-file: /root/.cloudflared/5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.json

ingress:
  - service: http://localhost:8000
EOF

echo "✅ Config created"
echo ""

# Show config
echo "📋 Config contents:"
cat ~/.cloudflared/config.yml
echo ""
echo "==========================================="
echo ""

# Check if credentials file exists
echo "🔍 Checking for credentials file..."
if [ -f /root/.cloudflared/5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.json ]; then
    echo "✅ Credentials file exists"
else
    echo "⚠️  Credentials file NOT found!"
    echo ""
    echo "You need to create it. Run this command with your token:"
    echo ""
    echo "cloudflared tunnel login"
    echo ""
    echo "Or if you have the token, run:"
    echo ""
    echo "cloudflared tunnel run --token <YOUR_TOKEN>"
    echo ""
    exit 1
fi
echo ""

# Stop any existing tunnel
echo "🛑 Stopping any existing tunnels..."
pkill -f cloudflared
sleep 3
echo "✅ Stopped"
echo ""

# Start tunnel
echo "🚀 Starting tunnel with new config..."
nohup cloudflared tunnel --config ~/.cloudflared/config.yml run 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61 > /workspace/logs/cloudflare-tunnel.log 2>&1 &
NEW_PID=$!
echo $NEW_PID > /workspace/cloudflare-tunnel.pid
echo "✅ Tunnel started (PID: $NEW_PID)"
echo ""

# Wait for tunnel
echo "⏳ Waiting 30 seconds for tunnel to initialize..."
sleep 30

# Check if running
if ps -p $NEW_PID > /dev/null 2>&1; then
    echo "✅ Tunnel is running!"
else
    echo "❌ Tunnel failed to start!"
    echo ""
    echo "📋 Logs:"
    tail -20 /workspace/logs/cloudflare-tunnel.log
    exit 1
fi
echo ""

# Test
echo "🧪 Testing direct URL..."
curl https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com/health
echo ""
echo ""
echo "==========================================="
echo "✅ Setup complete!"
echo "==========================================="
echo ""
echo "Your URL: https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com"
echo ""
