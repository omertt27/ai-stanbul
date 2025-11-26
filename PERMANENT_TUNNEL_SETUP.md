# Permanent Tunnel Setup Guide 🌐

**Date:** November 26, 2025  
**Purpose:** Set up a permanent tunnel for RunPod LLM server

---

## Why Permanent Tunnel?

Temporary Cloudflare tunnels (`trycloudflare.com`) have issues:
- ❌ URL changes every restart
- ❌ Need to update Render env vars constantly
- ❌ Not reliable for production
- ❌ Rate limited

Permanent tunnels solve this:
- ✅ Static URL (or stable across restarts)
- ✅ Set once in Render, forget it
- ✅ Production-ready
- ✅ Better performance

---

## Option 1: ngrok (Recommended) ⭐

### Why ngrok?
- Easy setup (5 minutes)
- Free tier available
- Reliable and fast
- Good documentation
- Paid tier ($8/mo) gives custom domain

### Setup Steps:

#### 1. Create ngrok Account (2 min)
1. Go to: https://dashboard.ngrok.com/signup
2. Sign up with email or GitHub
3. Verify email
4. Get your authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken

#### 2. Install on RunPod (3 min)

SSH into your RunPod pod and run:

```bash
# Download ngrok
cd /workspace
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz

# Extract
tar xvzf ngrok-v3-stable-linux-amd64.tgz

# Move to system path
sudo mv ngrok /usr/local/bin/

# Make executable
sudo chmod +x /usr/local/bin/ngrok

# Configure with your auth token (replace with your actual token)
ngrok config add-authtoken YOUR_AUTH_TOKEN_HERE

# Verify installation
ngrok --version
```

#### 3. Start Tunnel (1 min)

```bash
# Make sure your LLM server is running on port 8000
# If not, start it first:
# cd /workspace && python vllm_server_fixed.py &

# Start ngrok tunnel
nohup ngrok http 8000 > /workspace/ngrok.log 2>&1 &

# Check it's running
ps aux | grep ngrok
```

#### 4. Get Your URL (1 min)

```bash
# Get the public URL
curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['tunnels'][0]['public_url'] if data['tunnels'] else 'No tunnel running')"
```

**Example output:** `https://abc-123-def.ngrok-free.app`

**Your LLM_API_URL:** `https://abc-123-def.ngrok-free.app/v1` (add `/v1`!)

#### 5. Test Your Tunnel

```bash
# Test from RunPod
curl https://YOUR_NGROK_URL/v1/models

# Test from your local machine
curl https://YOUR_NGROK_URL/v1/models

# Both should return model list
```

### ngrok Free vs Paid:

| Feature | Free | Paid ($8/mo) |
|---------|------|--------------|
| Static URL | No (changes on restart) | Yes (custom domain) |
| Concurrent tunnels | 1 | 3 |
| Requests/min | 40 | Unlimited |
| Best for | Development | Production |

**Recommendation:** Start with free, upgrade if needed.

### Auto-restart ngrok Script

Create this script to auto-restart ngrok on RunPod boot:

```bash
# Create startup script
cat > /workspace/start_ngrok.sh << 'EOF'
#!/bin/bash
# Start ngrok tunnel on boot

# Wait for network
sleep 5

# Kill existing ngrok
pkill ngrok

# Start ngrok
cd /workspace
nohup ngrok http 8000 > /workspace/ngrok.log 2>&1 &

# Wait and get URL
sleep 10
curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; data=json.load(sys.stdin); print('Ngrok URL:', data['tunnels'][0]['public_url'] if data.get('tunnels') else 'Not ready yet')"
EOF

chmod +x /workspace/start_ngrok.sh

# Test it
/workspace/start_ngrok.sh
```

---

## Option 2: RunPod Public Port (Easiest if available)

### Check if Available:

1. Go to RunPod dashboard: https://www.runpod.io/console/pods
2. Click on your pod
3. Look for "TCP Port Mappings" or "Exposed Ports" section
4. If you see it, you can map internal port 8000

### Setup:

1. In RunPod dashboard:
   - Add TCP port mapping: `8000 → Public`
   - Save

2. Your URL will be:
   ```
   https://[pod-id]-8000.proxy.runpod.net
   ```

3. Test:
   ```bash
   curl https://[pod-id]-8000.proxy.runpod.net/v1/models
   ```

**Note:** Not all RunPod templates support this. If you don't see it, use ngrok.

---

## Option 3: Cloudflare Tunnel (Named) - Requires Domain

### Requirements:
- Your own domain (e.g., `mydomain.com`)
- Domain managed by Cloudflare (free)

### Setup:

#### 1. Prepare Domain (5 min)
1. Go to: https://dash.cloudflare.com/
2. Add your domain to Cloudflare
3. Update nameservers at your registrar

#### 2. Install cloudflared on RunPod (3 min)

```bash
# SSH into RunPod
cd /workspace

# Download cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64

# Install
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
sudo chmod +x /usr/local/bin/cloudflared

# Verify
cloudflared --version
```

#### 3. Authenticate (2 min)

```bash
# Login to Cloudflare
cloudflared tunnel login

# This opens browser - login and select domain
```

#### 4. Create Named Tunnel (3 min)

```bash
# Create tunnel
cloudflared tunnel create ai-istanbul-llm

# Note the tunnel ID from output
# Create DNS route
cloudflared tunnel route dns ai-istanbul-llm llm.yourdomain.com
```

#### 5. Configure Tunnel (2 min)

```bash
# Create config file
mkdir -p ~/.cloudflared
cat > ~/.cloudflared/config.yml << EOF
tunnel: ai-istanbul-llm
credentials-file: /root/.cloudflared/[TUNNEL-ID].json

ingress:
  - hostname: llm.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
EOF
```

#### 6. Start Tunnel (1 min)

```bash
# Start in background
nohup cloudflared tunnel run ai-istanbul-llm > /workspace/cloudflared.log 2>&1 &

# Check logs
tail -f /workspace/cloudflared.log
```

**Your URL:** `https://llm.yourdomain.com/v1`

---

## Recommended: Quick Start with ngrok

**For fastest setup (10 minutes total):**

```bash
# 1. SSH into RunPod
# 2. Run these commands:

# Install ngrok
cd /workspace
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin/
ngrok config add-authtoken YOUR_TOKEN_FROM_NGROK_DASHBOARD

# Start tunnel
nohup ngrok http 8000 > ngrok.log 2>&1 &

# Get URL
sleep 5
curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; data=json.load(sys.stdin); print('Your LLM URL:', data['tunnels'][0]['public_url'] + '/v1')"

# 3. Copy that URL
# 4. Go to Render → Environment → Set LLM_API_URL to that URL
# 5. Done!
```

---

## Set URL in Render

Once you have your permanent URL:

1. **Go to Render:** https://dashboard.render.com/
2. **Select backend service**
3. **Environment tab**
4. **Add/Update:**
   ```
   LLM_API_URL=https://your-permanent-url/v1
   PURE_LLM_MODE=true
   LLM_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
   ```
5. **Save Changes**
6. **Manual Deploy**

---

## Verification

### Test Tunnel:
```bash
# From anywhere (local machine, Render, etc.)
curl https://YOUR_TUNNEL_URL/v1/models

# Should return:
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.1-8B-Instruct",
      ...
    }
  ]
}
```

### Test LLM Generation:
```bash
curl -X POST https://YOUR_TUNNEL_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 50
  }'

# Should return AI response
```

---

## Troubleshooting

### ngrok: "ERR_NGROK_108"
**Issue:** Authtoken not configured  
**Fix:** Run `ngrok config add-authtoken YOUR_TOKEN`

### ngrok: "tunnel not found"
**Issue:** ngrok not running  
**Fix:** 
```bash
pkill ngrok
nohup ngrok http 8000 > ngrok.log 2>&1 &
```

### "Connection refused"
**Issue:** LLM server not running  
**Fix:**
```bash
cd /workspace
python vllm_server_fixed.py &
```

### URL changes after restart
**Issue:** Using ngrok free tier  
**Fix:** Either:
- Note new URL and update Render env vars
- Upgrade to ngrok paid ($8/mo) for static domain

### Cloudflare: "tunnel not found"
**Issue:** Config file wrong  
**Fix:** Check `~/.cloudflared/config.yml` has correct tunnel ID

---

## Production Best Practices

### For Development/Testing:
- ✅ ngrok free tier
- ✅ Update Render when URL changes
- ✅ Monitor ngrok dashboard

### For Production:
- ✅ ngrok paid ($8/mo) - Custom domain
- OR ✅ Cloudflare tunnel - Free with your domain
- ✅ Set up monitoring/alerts
- ✅ Auto-restart scripts

---

## Cost Comparison

| Option | Setup | Monthly | Best For |
|--------|-------|---------|----------|
| ngrok Free | 5 min | $0 | Development |
| ngrok Paid | 5 min | $8 | Production |
| Cloudflare | 15 min | $0 | Production (own domain) |
| RunPod Ports | 2 min | $0 | If available |

**Recommendation:** 
- Start: ngrok free
- Production: ngrok paid OR Cloudflare tunnel

---

## Next Steps

1. **Choose your option** (ngrok recommended)
2. **Set up tunnel** (follow steps above)
3. **Get permanent URL**
4. **Update Render environment variables**
5. **Deploy and test**
6. **Done! No more URL changes** ✅

---

## Quick Commands Reference

### ngrok:
```bash
# Install
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz && tar xvzf ngrok-*.tgz && sudo mv ngrok /usr/local/bin/

# Configure
ngrok config add-authtoken YOUR_TOKEN

# Start
nohup ngrok http 8000 > ngrok.log 2>&1 &

# Get URL
curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

### Cloudflare:
```bash
# Install
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared && sudo chmod +x /usr/local/bin/cloudflared

# Create tunnel
cloudflared tunnel create my-tunnel

# Run
nohup cloudflared tunnel --url http://localhost:8000 run my-tunnel > cloudflared.log 2>&1 &
```

---

**Ready?** Start with ngrok - it's the easiest! 🚀
