# 🚀 CLOUDFLARE TUNNEL WITH TOKEN (EASIEST METHOD)

## ✅ You Already Created a Tunnel in Cloudflare Dashboard!

This is the **easiest way** - Cloudflare gives you a token that contains everything needed.

---

## 🎯 Quick Setup (3 Commands)

### Step 1: SSH into RunPod

```bash
ssh -p 22003 -i ~/.ssh/id_ed25519 root@194.68.245.153
```

---

### Step 2: Install cloudflared

```bash
cd /workspace
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
```

---

### Step 3: Install Service with Your Token

Replace `eyJhIjoiYW...` with your actual token from Cloudflare:

```bash
cloudflared service install eyJhIjoiYW...YOUR_FULL_TOKEN_HERE...
```

⚠️ **IMPORTANT**: Copy your FULL token from the Cloudflare dashboard!

---

## 🎯 Complete One-Command Setup

Run this on RunPod terminal (replace `YOUR_TOKEN` with your actual token):

```bash
cd /workspace && \
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && \
chmod +x cloudflared-linux-amd64 && \
mv cloudflared-linux-amd64 /usr/local/bin/cloudflared && \
cloudflared service install eyJhIjoiYW...YOUR_FULL_TOKEN_HERE...
```

---

## 📋 What This Does

The token contains:
- ✅ Tunnel ID
- ✅ Tunnel credentials
- ✅ Configuration
- ✅ Everything needed!

**No manual config file needed!**

---

## 🌐 Configure Your Public URL

### In Cloudflare Dashboard:

1. Go to **Zero Trust** → **Networks** → **Tunnels**
2. Click on your tunnel name
3. Go to **Public Hostname** tab
4. Click **Add a public hostname**

**Configure:**
- **Subdomain**: `llm` (or any name you want)
- **Domain**: Select your domain
- **Service Type**: `HTTP`
- **URL**: `localhost:8000`

**Result**: `https://llm.yourdomain.com` → `http://localhost:8000`

---

## 🚀 Alternative: Start Without Service (nohup)

If you don't want to install as a system service, use nohup:

```bash
cd /workspace
mkdir -p logs

# Start tunnel with token
nohup cloudflared tunnel --no-autoupdate run --token eyJhIjoiYW...YOUR_TOKEN... \
  > /workspace/logs/cloudflare-tunnel.log 2>&1 &

# Save PID
echo $! > /workspace/cloudflare-tunnel.pid

# Disown
disown

echo "✅ Tunnel started with nohup!"
```

---

## 🎯 Complete nohup Command (Copy & Paste)

Replace `YOUR_TOKEN` with your actual token:

```bash
cd /workspace && \
mkdir -p logs && \
nohup cloudflared tunnel --no-autoupdate run --token eyJhIjoiYW...YOUR_TOKEN... \
  > /workspace/logs/cloudflare-tunnel.log 2>&1 & \
echo $! > /workspace/cloudflare-tunnel.pid && \
disown && \
echo "✅ Tunnel started!" && \
echo "" && \
echo "View logs: tail -f /workspace/logs/cloudflare-tunnel.log" && \
echo "Stop: kill \$(cat /workspace/cloudflare-tunnel.pid)"
```

---

## 🧪 Test Your Tunnel

### From RunPod:
```bash
# Test health (replace with your actual URL)
curl https://llm.yourdomain.com/health | python3 -m json.tool

# Test completion
curl -X POST https://llm.yourdomain.com/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Istanbul is", "max_tokens": 30}' \
  | python3 -m json.tool
```

### From Your Local Machine:
```bash
# Replace with your actual URL
export LLM_URL="https://llm.yourdomain.com"

# Health
curl $LLM_URL/health

# Completion
curl -X POST $LLM_URL/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me about Istanbul", "max_tokens": 50}'
```

---

## 🛠️ Management

### View Logs:
```bash
tail -f /workspace/logs/cloudflare-tunnel.log
```

### Check Status:
```bash
ps aux | grep cloudflared
```

### Stop Tunnel:
```bash
kill $(cat /workspace/cloudflare-tunnel.pid)
```

### Restart:
```bash
# Just run the nohup command again
```

### If Installed as Service:
```bash
# Start
sudo systemctl start cloudflared

# Stop
sudo systemctl stop cloudflared

# Restart
sudo systemctl restart cloudflared

# Status
sudo systemctl status cloudflared

# Enable auto-start on boot
sudo systemctl enable cloudflared

# View logs
sudo journalctl -u cloudflared -f
```

---

## 📊 Cloudflare Dashboard Configuration

### 1. Public Hostname Setup:

In Cloudflare Dashboard → Your Tunnel → Public Hostname:

```
Subdomain: llm
Domain: yourdomain.com
Service: 
  Type: HTTP
  URL: localhost:8000
```

**Result**: `https://llm.yourdomain.com` points to your LLM server!

### 2. Multiple Endpoints (Optional):

You can add more:

```
llm.yourdomain.com → localhost:8000  (LLM server)
api.yourdomain.com → localhost:8000  (Same server, different domain)
```

---

## ✅ Advantages of Token Method

| Feature | Manual Setup | Token Method |
|---------|-------------|--------------|
| Setup Steps | 8 steps | 3 steps |
| Login Required | Yes | No |
| Config File | Manual | Auto |
| Tunnel Creation | Manual | Via Dashboard |
| Best For | Advanced | Quick Start |

---

## 🎯 Backend Integration

Once tunnel is running:

### Update Backend .env:
```bash
LLM_SERVER_URL=https://llm.yourdomain.com
```

### Test from Backend:
```python
import requests

# Health
response = requests.get("https://llm.yourdomain.com/health")
print(response.json())

# Completion
response = requests.post(
    "https://llm.yourdomain.com/v1/completions",
    json={"prompt": "Istanbul is", "max_tokens": 50}
)
print(response.json()["text"])
```

---

## 📋 Complete Setup Checklist

- [ ] Install cloudflared on RunPod
- [ ] Copy your token from Cloudflare dashboard
- [ ] Run service install command with token
- [ ] Configure public hostname in Cloudflare dashboard
- [ ] Test health endpoint
- [ ] Test completion endpoint
- [ ] Update backend configuration

---

## 🚀 Quick Start Commands Summary

```bash
# 1. SSH to RunPod
ssh -p 22003 -i ~/.ssh/id_ed25519 root@194.68.245.153

# 2. Install cloudflared
cd /workspace
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# 3. Start with token (nohup method - replace YOUR_TOKEN)
nohup cloudflared tunnel --no-autoupdate run --token eyJhIjoiYW...YOUR_TOKEN... \
  > /workspace/logs/cloudflare-tunnel.log 2>&1 &

echo $! > /workspace/cloudflare-tunnel.pid
disown
```

---

## 🎉 That's It!

**Your permanent tunnel with a fixed URL is ready!**

**URL**: `https://llm.yourdomain.com` (or whatever you configured)

---

## 📞 Need Help?

### Where to find your token:
1. Go to Cloudflare Dashboard
2. Zero Trust → Networks → Tunnels
3. Click your tunnel
4. Click "Configure"
5. Copy the token from the install command

### Where to configure public hostname:
1. Same tunnel page
2. Click "Public Hostname" tab
3. Add hostname with `localhost:8000`

---

**Use the nohup method for RunPod - it's simpler and works perfectly!** 🚀
