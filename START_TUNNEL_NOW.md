# 🚀 CLOUDFLARE TUNNEL - READY TO START!

## ✅ Your Tunnel Token Retrieved!

**Tunnel Name**: LLM  
**Tunnel ID**: `5887803e-cf72-4fcc-82ce-4cc1f4b1dd61`  
**Token**: Retrieved successfully from your setup

---

## 🎯 START YOUR TUNNEL NOW (Copy & Paste)

Run this command on RunPod to start your Cloudflare tunnel:

```bash
cd /workspace && \
mkdir -p logs && \
nohup cloudflared tunnel --no-autoupdate run --token eyJhIjoiYWU3MGQ3ZDlmMTI2ZWM3MjAxYjkyMzNjNDNlZTI0NDEiLCJ0IjoiNTg4NzgwM2UtY2Y3Mi00ZmNjLTgyY2UtNGNjMWY0YjFkZDYxIiwicyI6Ik1EWmlOalZtWW1RdFpHUTVOaTAwTmpFNUxXRmlZMk10WW1FNU1HUTBOR1ZrWm1ZeSJ9 \
  > /workspace/logs/cloudflare-tunnel.log 2>&1 & \
echo $! > /workspace/cloudflare-tunnel.pid && \
disown && \
echo "" && \
echo "✅ Cloudflare Tunnel started!" && \
echo "PID: $(cat /workspace/cloudflare-tunnel.pid)" && \
echo "" && \
echo "View logs: tail -f /workspace/logs/cloudflare-tunnel.log" && \
echo "Check status: ps aux | grep cloudflared"
```

---

## 📋 Verify It's Running

```bash
# Check process
ps aux | grep cloudflared

# View logs (look for "Connection registered")
tail -f /workspace/logs/cloudflare-tunnel.log
```

**Look for**: `Connection <UUID> registered` - this means it's connected! ✅

Press `Ctrl+C` to exit log viewer.

---

## 🌐 Configure Public Hostname (REQUIRED)

Now configure how to access your tunnel:

### In Cloudflare Dashboard:

1. Go to: https://one.dash.cloudflare.com/
2. Navigate: **Zero Trust** → **Networks** → **Tunnels**
3. Click on tunnel: **LLM**
4. Go to **"Public Hostname"** tab
5. Click **"Add a public hostname"**

### Configuration:

```
Subdomain: llm
Domain: [Select your domain from dropdown]
Service:
  Type: HTTP
  URL: localhost:8000
```

**Save** and you're done!

This creates: `https://llm.yourdomain.com` → `http://localhost:8000`

---

## 🧪 Test External Access

Replace `llm.yourdomain.com` with your actual domain:

```bash
# Health check
curl https://llm.yourdomain.com/health

# Completion test
curl -X POST https://llm.yourdomain.com/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Istanbul is", "max_tokens": 30}'
```

---

## 🛠️ Management Commands

### View Logs
```bash
tail -f /workspace/logs/cloudflare-tunnel.log
```

### Check if Running
```bash
ps aux | grep cloudflared
```

### Stop Tunnel
```bash
kill $(cat /workspace/cloudflare-tunnel.pid)
```

### Restart Tunnel
```bash
# Stop first
kill $(cat /workspace/cloudflare-tunnel.pid)

# Then run the start command again (see above)
```

---

## 📊 What to Expect

### In Tunnel Logs:

```
2025-12-03T12:05:00Z INF Starting tunnel...
2025-12-03T12:05:01Z INF Connection <UUID> registered connIndex=0
2025-12-03T12:05:01Z INF Connection <UUID> registered connIndex=1
2025-12-03T12:05:01Z INF Connection <UUID> registered connIndex=2
2025-12-03T12:05:01Z INF Connection <UUID> registered connIndex=3
```

**4 connections registered = Tunnel is healthy!** ✅

### In Cloudflare Dashboard:

- **Status**: 🟢 Healthy (green)
- **Uptime**: Shows active time
- **Connectors**: 1 active

---

## 🚀 Complete Setup Checklist

- [x] Cloudflare tunnel "LLM" created
- [x] Token retrieved
- [x] cloudflared installed on RunPod
- [ ] **Tunnel started with command above** ⬅️ **DO THIS NOW**
- [ ] Verify logs show "Connection registered"
- [ ] Configure public hostname in Cloudflare dashboard
- [ ] Test external access with curl
- [ ] Update backend .env with tunnel URL

---

## 🎯 Quick Reference

| Item | Value |
|------|-------|
| Tunnel Name | LLM |
| Tunnel ID | 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61 |
| Local Service | http://localhost:8000 |
| Log File | /workspace/logs/cloudflare-tunnel.log |
| PID File | /workspace/cloudflare-tunnel.pid |

---

## 💡 Troubleshooting

### Tunnel Won't Start

```bash
# Check if cloudflared exists
which cloudflared

# Try running in foreground to see errors
cloudflared tunnel run --token eyJhIjoiYWU3MGQ3ZDlmMTI2ZWM3MjAxYjkyMzNjNDNlZTI0NDEiLCJ0IjoiNTg4NzgwM2UtY2Y3Mi00ZmNjLTgyY2UtNGNjMWY0YjFkZDYxIiwicyI6Ik1EWmlOalZtWW1RdFpHUTVOaTAwTmpFNUxXRmlZMk10WW1FNU1HUTBOR1ZrWm1ZeSJ9
```

### Can't Access Externally

1. Check tunnel is running: `ps aux | grep cloudflared`
2. Check logs: `tail -f /workspace/logs/cloudflare-tunnel.log`
3. Verify public hostname is configured in dashboard
4. Test LLM server locally: `curl http://localhost:8000/health`

### Service Already Installed Error

**Ignore it!** We don't need the service. Just use the nohup command above.

---

## 🎉 Next Steps

1. **Run the start command above** on RunPod
2. **Check logs** for "Connection registered"
3. **Configure public hostname** in Cloudflare dashboard
4. **Test external access** with your domain
5. **Update backend .env** with your tunnel URL

---

**Your tunnel is ready to start! Copy the command above and run it on RunPod.** 🚀
