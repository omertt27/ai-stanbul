# 🔧 Fix Direct Tunnel Access (Deleted Routes)

**Date:** December 3, 2025  
**Issue:** Direct `.cfargotunnel.com` URL not working after deleting hostname routes  
**Solution:** Update tunnel config with catch-all ingress rule

---

## 🎯 What Happened

When you deleted the hostname routes in Cloudflare dashboard, the tunnel lost its routing rules. Now you need to configure it to accept traffic from the direct `.cfargotunnel.com` URL.

---

## ✅ Quick Fix (Run on RunPod)

### Step 1: Check Current Config

```bash
cat ~/.cloudflared/config.yml
```

You should see something like:
```yaml
tunnel: 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61
credentials-file: /root/.cloudflared/5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.json
```

### Step 2: Update Config with Catch-All Rule

```bash
# Backup current config
cp ~/.cloudflared/config.yml ~/.cloudflared/config.yml.backup

# Create new config
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61
credentials-file: /root/.cloudflared/5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.json

ingress:
  - service: http://localhost:8000
EOF
```

**What this does:**
- The single `service` line is a **catch-all rule**
- It routes ALL traffic (including `.cfargotunnel.com`) to your LLM server
- Simple and effective!

### Step 3: Restart Tunnel

```bash
# Stop old tunnel
kill $(cat /workspace/cloudflare-tunnel.pid)

# Wait a moment
sleep 3

# Start with new config
nohup cloudflared tunnel --config ~/.cloudflared/config.yml run 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61 > /workspace/logs/cloudflare-tunnel.log 2>&1 &

# Save new PID
echo $! > /workspace/cloudflare-tunnel.pid

echo "✅ Tunnel restarted!"
```

### Step 4: Wait and Test

```bash
# Wait 30 seconds for tunnel to register
echo "Waiting 30 seconds for tunnel to initialize..."
sleep 30

# Test direct URL
curl https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "device": "cuda"
}
```

---

## 🔍 Verify Everything

### 1. Check Tunnel Process
```bash
ps aux | grep cloudflared
```
Should show running process.

### 2. Check Tunnel Logs
```bash
tail -20 /workspace/logs/cloudflare-tunnel.log
```
Should show "Registered tunnel connection" messages (4 of them).

### 3. Check Local Server
```bash
curl http://localhost:8000/health
```
Should return healthy status.

### 4. Test Direct URL
```bash
curl https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com/health
```
Should return healthy status.

---

## 🎯 Your Direct Tunnel URL

```
https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com
```

**Use this in:**
- Render backend: `LLM_SERVER_URL`
- Vercel frontend: `VITE_LLM_SERVER_URL`

---

## 🧪 Full Test Commands

```bash
# Health check
curl https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com/health

# Test generation
curl -X POST https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Say hello in Turkish",
    "max_tokens": 50,
    "temperature": 0.7
  }'

# Test streaming (should see data streaming)
curl -N https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Count to 5",
    "max_tokens": 30
  }'
```

---

## 🛠️ Troubleshooting

### Still Getting "Couldn't connect to server"?

**1. Check tunnel is running:**
```bash
ps aux | grep cloudflared
```

**2. Check for errors in logs:**
```bash
grep -i error /workspace/logs/cloudflare-tunnel.log
```

**3. Verify config syntax:**
```bash
cat ~/.cloudflared/config.yml
```

**4. Check LLM server is running:**
```bash
ps aux | grep llm_server
curl http://localhost:8000/health
```

### Getting 502 Bad Gateway?

**Cause:** LLM server not responding.

**Fix:**
```bash
# Check if server is running
ps aux | grep llm_server

# If not, start it
cd /workspace
nohup python llm_server.py > /workspace/logs/llm_server.log 2>&1 &
echo $! > /workspace/llm_server.pid
```

### Getting 404 Not Found?

**Cause:** Wrong endpoint path.

**Fix:** Make sure you're using correct paths:
- `/health` - Health check
- `/generate` - Text generation
- `/stream` - Streaming generation

---

## 📋 Complete Setup Commands (Copy-Paste)

Run all of these on RunPod in order:

```bash
# 1. Backup current config
cp ~/.cloudflared/config.yml ~/.cloudflared/config.yml.backup

# 2. Create new config with catch-all rule
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61
credentials-file: /root/.cloudflared/5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.json

ingress:
  - service: http://localhost:8000
EOF

# 3. Stop old tunnel
kill $(cat /workspace/cloudflare-tunnel.pid) 2>/dev/null
sleep 3

# 4. Start new tunnel
nohup cloudflared tunnel --config ~/.cloudflared/config.yml run 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61 > /workspace/logs/cloudflare-tunnel.log 2>&1 &
echo $! > /workspace/cloudflare-tunnel.pid

# 5. Wait for initialization
echo "Waiting 30 seconds for tunnel to register..."
sleep 30

# 6. Test direct URL
curl https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com/health
```

---

## ✅ Success Indicators

When it's working, you'll see:

1. **Tunnel logs show connections:**
   ```
   Registered tunnel connection connIndex=0
   Registered tunnel connection connIndex=1
   Registered tunnel connection connIndex=2
   Registered tunnel connection connIndex=3
   ```

2. **Local server responds:**
   ```bash
   curl http://localhost:8000/health
   # Returns: {"status":"healthy",...}
   ```

3. **Direct URL responds:**
   ```bash
   curl https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com/health
   # Returns: {"status":"healthy",...}
   ```

---

## 🎉 Next Steps After It Works

1. ✅ **Update Backend** (Render):
   - Add env var: `LLM_SERVER_URL=https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com`

2. ✅ **Update Frontend** (Vercel):
   - Add env var: `VITE_LLM_SERVER_URL=https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com`

3. ✅ **Test Integration**:
   - Open https://aistanbul.net
   - Try the chat feature
   - Check browser console for errors

---

## 📝 Understanding the Config

### Minimal Config (What you need now):
```yaml
tunnel: 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61
credentials-file: /root/.cloudflared/5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.json

ingress:
  - service: http://localhost:8000
```

### With Custom Domain (Add later):
```yaml
tunnel: 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61
credentials-file: /root/.cloudflared/5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.json

ingress:
  - hostname: llm.aistanbul.net
    service: http://localhost:8000
  - service: http://localhost:8000
```

**Difference:**
- First rule: Routes `llm.aistanbul.net` → LLM server
- Second rule: Routes everything else (including `.cfargotunnel.com`) → LLM server

---

## 🔒 Security Notes

✅ **Your direct URL is secure:**
- HTTPS with Cloudflare SSL certificate
- Encrypted tunnel connection
- Only accessible via the specific URL
- Not listed publicly

⚠️ **Consider adding authentication** if needed:
- Add API keys to your LLM server
- Use Cloudflare Access for additional protection
- Configure rate limiting in Cloudflare

---

**Last Updated:** December 3, 2025  
**Status:** Ready to fix  
**Time Required:** 5 minutes  
**Next:** Run the commands above on RunPod
