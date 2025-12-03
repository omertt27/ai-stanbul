# 🚀 CLOUDFLARE TUNNEL SERVICE - MANAGEMENT GUIDE

## ✅ Service Successfully Installed!

Your Cloudflare Tunnel is now installed as a Linux systemd service.

---

## 🎯 Quick Start Commands

### Start the Service

```bash
sudo systemctl start cloudflared
```

### Check Status

```bash
sudo systemctl status cloudflared
```

Expected output:
```
● cloudflared.service - cloudflared
     Loaded: loaded (/etc/systemd/system/cloudflared.service; enabled)
     Active: active (running) since...
```

### View Logs

```bash
# Real-time logs
sudo journalctl -u cloudflared -f

# Last 50 lines
sudo journalctl -u cloudflared -n 50

# All logs
sudo journalctl -u cloudflared
```

---

## 🛠️ Complete Service Management

### Start Service
```bash
sudo systemctl start cloudflared
```

### Stop Service
```bash
sudo systemctl stop cloudflared
```

### Restart Service
```bash
sudo systemctl restart cloudflared
```

### Check Status
```bash
sudo systemctl status cloudflared
```

### Enable Auto-Start on Boot
```bash
sudo systemctl enable cloudflared
```

### Disable Auto-Start on Boot
```bash
sudo systemctl disable cloudflared
```

---

## 📋 Next Steps

### 1️⃣ Start the Service

```bash
sudo systemctl start cloudflared
```

### 2️⃣ Configure Public Hostname in Cloudflare Dashboard

1. Go to **Cloudflare Dashboard** → **Zero Trust** → **Networks** → **Tunnels**
2. Click on your tunnel name
3. Go to **Public Hostname** tab
4. Click **Add a public hostname**

**Configure:**
- **Subdomain**: `llm` (or any name you want)
- **Domain**: Select your domain
- **Service Type**: `HTTP`
- **URL**: `localhost:8000`

**Result**: `https://llm.yourdomain.com` → `http://localhost:8000`

### 3️⃣ Verify Service is Running

```bash
# Check status
sudo systemctl status cloudflared

# Check logs for "Connection established"
sudo journalctl -u cloudflared -n 20
```

Look for lines like:
```
Connection <UUID> registered
```

### 4️⃣ Test the Tunnel

```bash
# Replace with your actual URL
curl https://llm.yourdomain.com/health

# Test completion
curl -X POST https://llm.yourdomain.com/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Istanbul is", "max_tokens": 30}'
```

---

## 🔍 Troubleshooting

### Service Won't Start

```bash
# Check logs for errors
sudo journalctl -u cloudflared -n 50

# Check service file
sudo systemctl cat cloudflared

# Restart service
sudo systemctl restart cloudflared
```

### View Service Configuration

```bash
sudo systemctl cat cloudflared
```

### Check if Service is Enabled

```bash
sudo systemctl is-enabled cloudflared
```

### Check if Service is Active

```bash
sudo systemctl is-active cloudflared
```

---

## 📊 Service Status Codes

| Status | Meaning |
|--------|---------|
| `active (running)` | ✅ Service is running |
| `inactive (dead)` | ⚠️ Service is stopped |
| `failed` | ❌ Service crashed |
| `activating (start)` | 🔄 Service is starting |

---

## 🎯 Complete Startup Sequence

Run these commands on RunPod:

```bash
# 1. Start the service
sudo systemctl start cloudflared

# 2. Check status
sudo systemctl status cloudflared

# 3. Enable auto-start (optional)
sudo systemctl enable cloudflared

# 4. View logs
sudo journalctl -u cloudflared -f
```

Press `Ctrl+C` to exit the log viewer.

---

## 🧪 Test Everything

### Check LLM Server is Running

```bash
# Check if Python process is running
ps aux | grep llm_server.py

# Check if port 8000 is listening
ss -tlnp | grep 8000

# Test locally
curl http://localhost:8000/health
```

### Check Cloudflare Tunnel is Running

```bash
# Check service status
sudo systemctl status cloudflared

# Check logs
sudo journalctl -u cloudflared -n 20
```

### Test External Access

```bash
# Replace with your actual URL
curl https://llm.yourdomain.com/health
```

---

## 📋 Production Checklist

- [ ] Service installed successfully
- [ ] Service started: `sudo systemctl start cloudflared`
- [ ] Service status shows "active (running)"
- [ ] Public hostname configured in Cloudflare dashboard
- [ ] Logs show "Connection registered"
- [ ] Health endpoint accessible externally
- [ ] Completion endpoint working
- [ ] Auto-start enabled: `sudo systemctl enable cloudflared`

---

## 🚀 Backend Integration

Once tunnel is working:

### Update Backend .env

```bash
LLM_SERVER_URL=https://llm.yourdomain.com
```

### Test from Backend

```python
import requests

# Health check
response = requests.get("https://llm.yourdomain.com/health")
print(response.json())

# Completion test
response = requests.post(
    "https://llm.yourdomain.com/v1/completions",
    json={"prompt": "Istanbul is", "max_tokens": 50}
)
print(response.json()["text"])
```

---

## 🎉 Service Management Summary

```bash
# Start
sudo systemctl start cloudflared

# Stop
sudo systemctl stop cloudflared

# Restart
sudo systemctl restart cloudflared

# Status
sudo systemctl status cloudflared

# Logs
sudo journalctl -u cloudflared -f

# Enable auto-start
sudo systemctl enable cloudflared
```

---

## ⚠️ Important Notes

1. **Use `systemctl` commands**, not `cloudflared service` commands
2. **Service runs automatically** after installation
3. **Check logs** if tunnel isn't working: `sudo journalctl -u cloudflared -f`
4. **Public hostname** must be configured in Cloudflare dashboard
5. **LLM server** must be running on localhost:8000

---

## 📞 Common Commands Reference

```bash
# Is service running?
sudo systemctl is-active cloudflared

# Is service enabled to start on boot?
sudo systemctl is-enabled cloudflared

# Show service file
sudo systemctl cat cloudflared

# Reload systemd after changes
sudo systemctl daemon-reload

# Restart service
sudo systemctl restart cloudflared

# View last 100 log lines
sudo journalctl -u cloudflared -n 100

# View logs since 10 minutes ago
sudo journalctl -u cloudflared --since "10 minutes ago"
```

---

**Your tunnel service is ready! Start it with `sudo systemctl start cloudflared`** 🚀
