# 🎯 Configure Cloudflare Tunnel Route via API

**Date:** December 3, 2025  
**Goal:** Enable production-ready `.cfargotunnel.com` URL

---

## 🔧 Method 1: Via Cloudflare API (Recommended)

Since the dashboard UI is difficult to navigate, we can use the Cloudflare API to configure the route.

### Step 1: Get Your Cloudflare API Token

1. Go to: https://dash.cloudflare.com/profile/api-tokens
2. Click: "Create Token"
3. **DO NOT USE "Edit Cloudflare Zero Trust"** - it doesn't have tunnel permissions!
4. Instead, use template: **"Edit Cloudflare Tunnels"** (if available)
5. Or create **Custom Token** with these permissions:
   - **Account Permissions:**
     - `Cloudflare Tunnel` → **Edit** ✅ (REQUIRED)
     - `Account Settings` → **Read** (optional)
   - **Zone Permissions** (only if using custom domain):
     - `DNS` → **Edit**
     - `Zone` → **Read**
6. Copy the token

### Step 2: Configure Route via API

Run this on your local machine (or RunPod):

```bash
# Set your credentials
CLOUDFLARE_ACCOUNT_ID="ae70d7d9f126ec7201b9233c43ee2441"
CLOUDFLARE_API_TOKEN="YOUR_API_TOKEN_HERE"
TUNNEL_ID="3c9f3076-300f-4a61-b923-cf7be81e2919"

# Add ingress rule for direct tunnel URL
curl -X PUT \
  "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/cfd_tunnel/${TUNNEL_ID}/configurations" \
  -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
  -H "Content-Type: application/json" \
  --data '{
    "config": {
      "ingress": [
        {
          "service": "http://localhost:8000"
        }
      ]
    }
  }'
```

---

## 🔧 Method 2: Via Dashboard (Step-by-Step)

If you want to use the dashboard, here's a detailed guide:

### Option A: Via "Configure" Tab

1. **Go to:** https://one.dash.cloudflare.com/
2. **Click:** Zero Trust (in left sidebar)
3. **Click:** Networks → Tunnels
4. **Find:** Your tunnel (ID: `3c9f3076...`)
5. **Click:** The tunnel name or row
6. **Look for tabs at top:**
   - Overview
   - **Configure** ← Click this
   - Metrics
   - Logs

7. **In Configure tab:**
   - Look for "Public Hostname" section
   - Or "Ingress Rules" section
   - Click "Add a public hostname" or "Add rule"

8. **Fill in:**
   ```
   Subdomain: (leave empty for catch-all)
   Domain: (leave empty for .cfargotunnel.com)
   Path: (leave empty)
   Type: HTTP
   URL: localhost:8000
   ```

9. **Save**

### Option B: Via "Edit" Button

1. **Go to tunnels list**
2. **Click the three dots (...)** next to your tunnel
3. **Click "Configure"** or **"Edit"**
4. **Follow same steps as above**

### Option C: Create New Public Hostname

1. **In tunnel details page**
2. **Scroll down to "Public Hostname" section**
3. **Click "+ Add a public hostname"**
4. **Fill in as above**

---

## 🔧 Method 3: Recreate Tunnel with Config File (Most Reliable)

This creates everything fresh with a config file that ensures the route works:

### On RunPod:

```bash
# Stop current tunnel
pkill -f cloudflared
sleep 3

# Remove old config
rm -rf ~/.cloudflared/*

# Create new config with ingress rule
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: 3c9f3076-300f-4a61-b923-cf7be81e2919
credentials-file: /root/.cloudflared/cert.pem

ingress:
  - service: http://localhost:8000
EOF

# Login to Cloudflare (this creates cert.pem)
cloudflared tunnel login

# This will open a browser or give you a URL to visit
# Follow the instructions to authenticate
# After auth, cert.pem will be created

# Start tunnel with config
nohup cloudflared tunnel --config ~/.cloudflared/config.yml run 3c9f3076-300f-4a61-b923-cf7be81e2919 > /workspace/logs/cloudflare-tunnel.log 2>&1 &
echo $! > /workspace/cloudflare-tunnel.pid

# Wait and test
sleep 30
curl https://3c9f3076-300f-4a61-b923-cf7be81e2919.cfargotunnel.com/health
```

---

## 🔧 Method 4: CLI Configuration (Easiest!)

The Cloudflare CLI can configure routes directly:

```bash
# On RunPod or local machine

# Install cloudflared if needed
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# Configure the route
cloudflared tunnel route dns 3c9f3076-300f-4a61-b923-cf7be81e2919 http://localhost:8000

# Or for catch-all:
cloudflared tunnel ingress validate ~/.cloudflared/config.yml
```

---

## 🎯 Recommended: Use Method 1 (API)

The API method is cleanest. Here's the complete script:

### Create `configure_tunnel_route.sh`:

```bash
#!/bin/bash

echo "🔧 Configuring Cloudflare Tunnel Route via API"
echo "=============================================="
echo ""

# IMPORTANT: Replace with your actual API token
read -p "Enter your Cloudflare API Token: " API_TOKEN

ACCOUNT_ID="ae70d7d9f126ec7201b9233c43ee2441"
TUNNEL_ID="3c9f3076-300f-4a61-b923-cf7be81e2919"

echo ""
echo "📡 Sending API request to Cloudflare..."

RESPONSE=$(curl -s -X PUT \
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

echo ""
echo "📋 API Response:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"

if echo "$RESPONSE" | grep -q '"success":true'; then
    echo ""
    echo "✅ Route configured successfully!"
    echo ""
    echo "⏳ Waiting 30 seconds for propagation..."
    sleep 30
    
    echo ""
    echo "🧪 Testing direct URL..."
    curl https://3c9f3076-300f-4a61-b923-cf7be81e2919.cfargotunnel.com/health
    
    echo ""
    echo ""
    echo "=============================================="
    echo "✅ Your production URL is ready:"
    echo "https://3c9f3076-300f-4a61-b923-cf7be81e2919.cfargotunnel.com"
    echo "=============================================="
else
    echo ""
    echo "❌ Configuration failed. Check the response above."
    echo ""
    echo "💡 Alternative: Use RunPod URL for now:"
    echo "https://4r1su4zfuok0s7-19123.proxy.runpod.net/nsvmrgaqgp8j29z4dytki4habch6tazn"
fi
```

---

## 📋 Quick Start (Choose One)

### If you have API token:
```bash
# Run the API script
./configure_tunnel_route.sh
```

### If you prefer dashboard:
1. Navigate to tunnel in Cloudflare dashboard
2. Find "Configure" or "Public Hostname" section
3. Add route: HTTP → localhost:8000
4. Save

### If both fail:
Use RunPod URL temporarily:
```
https://4r1su4zfuok0s7-19123.proxy.runpod.net/nsvmrgaqgp8j29z4dytki4habch6tazn
```

---

## 🎉 After Configuration Success

Once the route is configured, update your environment variables:

### Render Backend:
```
LLM_SERVER_URL=https://3c9f3076-300f-4a61-b923-cf7be81e2919.cfargotunnel.com
```

### Vercel Frontend:
```
VITE_LLM_SERVER_URL=https://3c9f3076-300f-4a61-b923-cf7be81e2919.cfargotunnel.com
```

---

## 🔍 Troubleshooting

### API returns error?
- Check API token has correct permissions
- Verify account ID is correct
- Make sure tunnel is running

### Dashboard won't save?
- Try different browser
- Clear cache
- Try incognito mode

### URL still not working?
- Wait 2-3 minutes for DNS propagation
- Check tunnel is running: `ps aux | grep cloudflared`
- Check local server: `curl http://localhost:8000/health`
- View tunnel logs: `tail -f /workspace/logs/cloudflare-tunnel.log`

---

**Last Updated:** December 3, 2025  
**Next:** Get API token and run configuration script
