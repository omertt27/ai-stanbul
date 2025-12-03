# 🌐 PERMANENT CLOUDFLARE TUNNEL SETUP

## Why Permanent Tunnel?

✅ **Persistent URL** - Never changes  
✅ **Custom Domain** - Use your own domain  
✅ **Professional** - Production-ready  
✅ **Auto-reconnect** - Survives restarts  
✅ **Free** - No cost with Cloudflare account  

---

## Prerequisites

You need:
1. A Cloudflare account (free)
2. A domain in Cloudflare (free or use existing)
3. SSH access to RunPod

---

## 🚀 Step-by-Step Setup

### Step 1: SSH into RunPod

```bash
ssh -p 22003 -i ~/.ssh/id_ed25519 root@194.68.245.153
```

---

### Step 2: Install cloudflared

```bash
cd /workspace

# Download cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64

# Make executable
chmod +x cloudflared-linux-amd64

# Move to system bin
mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# Verify
cloudflared --version
```

---

### Step 3: Authenticate with Cloudflare

```bash
cloudflared tunnel login
```

**What happens:**
1. A URL appears in your terminal
2. Copy and paste it in your browser
3. Log in to Cloudflare
4. Select your domain
5. Click "Authorize"

**This creates**: `~/.cloudflared/cert.pem`

---

### Step 4: Create a Named Tunnel

```bash
cloudflared tunnel create llm-server
```

**Output will show:**
```
Tunnel credentials written to /root/.cloudflared/TUNNEL-ID.json
Created tunnel llm-server with id TUNNEL-ID
```

**📋 SAVE THE TUNNEL ID!** (e.g., `abc123-def456-ghi789-jkl012`)

---

### Step 5: Create Configuration File

```bash
# Get your tunnel ID from the previous step
TUNNEL_ID="your-tunnel-id-here"

# Create config file
cat > ~/.cloudflared/config.yml << EOF
tunnel: $TUNNEL_ID
credentials-file: /root/.cloudflared/${TUNNEL_ID}.json

ingress:
  - hostname: llm.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
EOF
```

**Replace:**
- `your-tunnel-id-here` with your actual tunnel ID
- `llm.yourdomain.com` with your desired subdomain

**Example:**
- If your domain is `example.com`
- Use `llm.example.com` or `api.example.com`

---

### Step 6: Route DNS to Tunnel

```bash
cloudflared tunnel route dns llm-server llm.yourdomain.com
```

**Replace** `llm.yourdomain.com` with your subdomain.

**This creates** a CNAME record in Cloudflare automatically!

---

### Step 7: Start Tunnel with nohup

```bash
cd /workspace
mkdir -p logs

# Start tunnel
nohup cloudflared tunnel run llm-server > /workspace/logs/cloudflare-tunnel.log 2>&1 &

# Save PID
echo $! > /workspace/cloudflare-tunnel.pid

# Disown
disown

echo "✅ Permanent tunnel started!"
```

---

### Step 8: Test Your Permanent URL

```bash
# Replace with your actual domain
curl https://llm.yourdomain.com/health | python3 -m json.tool

# Test completion
curl -X POST https://llm.yourdomain.com/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Istanbul is", "max_tokens": 30}' \
  | python3 -m json.tool
```

---

## 🎯 Complete Setup Script

Copy this entire script to RunPod terminal:

```bash
#!/bin/bash
# Permanent Cloudflare Tunnel Setup

echo "🌐 Setting up permanent Cloudflare Tunnel..."
echo ""

# Install cloudflared
if ! command -v cloudflared &> /dev/null; then
    echo "📥 Installing cloudflared..."
    cd /workspace
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x cloudflared-linux-amd64
    mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
    echo "✅ cloudflared installed!"
fi

# Check if already authenticated
if [ ! -f ~/.cloudflared/cert.pem ]; then
    echo ""
    echo "🔐 Please authenticate with Cloudflare:"
    echo "   1. Copy the URL that appears below"
    echo "   2. Open it in your browser"
    echo "   3. Log in and authorize"
    echo ""
    cloudflared tunnel login
    echo ""
fi

# List existing tunnels
echo "📋 Existing tunnels:"
cloudflared tunnel list

echo ""
echo "To create a new tunnel, run:"
echo "  cloudflared tunnel create llm-server"
echo ""
echo "Then note your tunnel ID and continue with the setup."
```

---

## 📝 Auto-Start Script

Create a script to start the tunnel automatically:

```bash
cat > /workspace/start_permanent_tunnel.sh << 'EOFSCRIPT'
#!/bin/bash
# Start Permanent Cloudflare Tunnel

cd /workspace

# Check if config exists
if [ ! -f ~/.cloudflared/config.yml ]; then
    echo "❌ Config file not found!"
    echo "Please create ~/.cloudflared/config.yml first"
    exit 1
fi

# Kill existing tunnel
pkill cloudflared 2>/dev/null || true
sleep 2

# Create logs directory
mkdir -p /workspace/logs

# Start tunnel
echo "🌐 Starting permanent Cloudflare Tunnel..."
nohup cloudflared tunnel run llm-server > /workspace/logs/cloudflare-tunnel.log 2>&1 &

# Save PID
echo $! > /workspace/cloudflare-tunnel.pid
disown

echo "✅ Tunnel started!"
echo ""
echo "View logs:"
echo "  tail -f /workspace/logs/cloudflare-tunnel.log"
echo ""
echo "Check status:"
echo "  cloudflared tunnel info llm-server"
EOFSCRIPT

chmod +x /workspace/start_permanent_tunnel.sh
```

---

## 🛠️ Management Commands

### Check Tunnel Status:
```bash
cloudflared tunnel info llm-server
```

### List All Tunnels:
```bash
cloudflared tunnel list
```

### View Logs:
```bash
tail -f /workspace/logs/cloudflare-tunnel.log
```

### Check Running Process:
```bash
ps aux | grep cloudflared
```

### Stop Tunnel:
```bash
kill $(cat /workspace/cloudflare-tunnel.pid)
```

### Restart Tunnel:
```bash
/workspace/start_permanent_tunnel.sh
```

### Delete Tunnel:
```bash
cloudflared tunnel delete llm-server
```

---

## 📋 Configuration File Example

**~/.cloudflared/config.yml:**

```yaml
tunnel: abc123-def456-ghi789
credentials-file: /root/.cloudflared/abc123-def456-ghi789.json

ingress:
  # Your LLM server
  - hostname: llm.yourdomain.com
    service: http://localhost:8000
  
  # Catch-all (required)
  - service: http_status:404
```

---

## 🌐 Using a Free Cloudflare Domain

If you don't have a domain:

### Option 1: Register Free Domain
1. Go to any domain registrar
2. Get a free `.tk`, `.ml`, `.ga`, etc.
3. Add it to Cloudflare

### Option 2: Use Subdomain
If you have a domain already in Cloudflare, just use a subdomain:
- `llm.example.com`
- `api.example.com`
- `ai.example.com`

---

## 🔒 Security Features

### 1. Enable HTTPS (Automatic)
Cloudflare automatically provides SSL/TLS certificate

### 2. Add Access Control (Optional)

In Cloudflare Dashboard:
1. Go to **Zero Trust** → **Access** → **Applications**
2. Click **Add an application**
3. Select **Self-hosted**
4. Add your tunnel URL
5. Configure authentication (email, Google, etc.)

### 3. Rate Limiting (Recommended)

In Cloudflare Dashboard:
1. Go to **Security** → **WAF**
2. Create rate limiting rule
3. Set limits (e.g., 100 requests per minute)

---

## 🎯 Backend Integration

Once your permanent tunnel is running:

### Update Backend .env:
```bash
LLM_SERVER_URL=https://llm.yourdomain.com
```

### Test from Backend:
```python
import requests

# Health check
response = requests.get("https://llm.yourdomain.com/health")
print(response.json())

# Completion
response = requests.post(
    "https://llm.yourdomain.com/v1/completions",
    json={
        "prompt": "What are the best restaurants in Istanbul?",
        "max_tokens": 100
    }
)
print(response.json()["text"])
```

---

## ✅ Advantages of Permanent Tunnel

| Feature | Quick Tunnel | Permanent Tunnel |
|---------|-------------|------------------|
| URL Stability | ❌ Changes | ✅ Fixed |
| Custom Domain | ❌ No | ✅ Yes |
| Auto-reconnect | ⚠️ Limited | ✅ Yes |
| Production Use | ❌ No | ✅ Yes |
| Free | ✅ Yes | ✅ Yes |
| Setup Time | 1 min | 5 min |

---

## 📞 Troubleshooting

### Tunnel not connecting?
```bash
# Check logs
tail -f /workspace/logs/cloudflare-tunnel.log

# Check config
cat ~/.cloudflared/config.yml

# Test connectivity
cloudflared tunnel info llm-server
```

### DNS not resolving?
- Wait 1-2 minutes for DNS propagation
- Check Cloudflare DNS settings
- Verify CNAME record exists

### Certificate issues?
```bash
# Re-authenticate
cloudflared tunnel login
```

---

## 🚀 Quick Checklist

- [ ] Install cloudflared
- [ ] Login to Cloudflare (`cloudflared tunnel login`)
- [ ] Create tunnel (`cloudflared tunnel create llm-server`)
- [ ] Save tunnel ID
- [ ] Create config file with your domain
- [ ] Route DNS (`cloudflared tunnel route dns ...`)
- [ ] Start tunnel with nohup
- [ ] Test endpoints
- [ ] Update backend configuration

---

**Your permanent tunnel will have a fixed URL that never changes!** 🌐

**Example**: `https://llm.yourdomain.com` ✨
