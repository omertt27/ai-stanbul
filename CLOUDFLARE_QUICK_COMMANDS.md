# 🚀 CLOUDFLARE TUNNEL - QUICK COMMANDS

## 📋 Copy & Paste This on RunPod Terminal

### One-Command Setup (Recommended):

```bash
cd /workspace && \
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && \
chmod +x cloudflared-linux-amd64 && \
mv cloudflared-linux-amd64 /usr/local/bin/cloudflared && \
mkdir -p logs && \
nohup cloudflared tunnel --url http://localhost:8000 > /workspace/logs/cloudflare-tunnel.log 2>&1 & \
echo $! > /workspace/cloudflare-tunnel.pid && \
disown && \
echo "✅ Tunnel starting... Wait 5 seconds..." && \
sleep 5 && \
echo "" && \
echo "📋 YOUR CLOUDFLARE URL:" && \
grep -o "https://.*trycloudflare.com" /workspace/logs/cloudflare-tunnel.log | head -1 && \
echo "" && \
echo "✅ Copy the URL above and use it in your backend!"
```

---

## 🔍 Get Your Tunnel URL

```bash
grep -o "https://.*trycloudflare.com" /workspace/logs/cloudflare-tunnel.log | head -1
```

---

## 🧪 Test from RunPod

```bash
# Get your URL first
URL=$(grep -o "https://.*trycloudflare.com" /workspace/logs/cloudflare-tunnel.log | head -1)

# Test health
curl $URL/health | python3 -m json.tool

# Test completion
curl -X POST $URL/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Istanbul is", "max_tokens": 30}' \
  | python3 -m json.tool
```

---

## 🛠️ Management

### Check Status:
```bash
ps aux | grep cloudflared
```

### View Logs:
```bash
tail -f /workspace/logs/cloudflare-tunnel.log
```

### Stop Tunnel:
```bash
kill $(cat /workspace/cloudflare-tunnel.pid)
```

### Restart:
```bash
cd /workspace && \
nohup cloudflared tunnel --url http://localhost:8000 > /workspace/logs/cloudflare-tunnel.log 2>&1 & \
echo $! > /workspace/cloudflare-tunnel.pid && \
disown && \
sleep 5 && \
grep -o "https://.*trycloudflare.com" /workspace/logs/cloudflare-tunnel.log | head -1
```

---

## 🎯 Backend Integration

Once you have your URL (e.g., `https://abc-123.trycloudflare.com`):

### Update `.env` file:
```bash
LLM_SERVER_URL=https://your-actual-url.trycloudflare.com
```

### Test from Backend:
```python
import requests

response = requests.get("https://your-url.trycloudflare.com/health")
print(response.json())
```

---

## ✅ What This Does

1. **Downloads** cloudflared
2. **Installs** it system-wide
3. **Starts** tunnel with nohup (runs in background)
4. **Saves** PID for management
5. **Disowns** process (survives SSH disconnect)
6. **Shows** your public URL

---

## 🌟 Benefits

✅ **Free** - No signup required  
✅ **Fast** - Instant setup  
✅ **Persistent** - Runs with nohup  
✅ **Public** - Accessible from anywhere  
✅ **Secure** - HTTPS by default  

---

**Just run the one-command setup and you're done!** 🚀
