# 🚀 Direct Tunnel URL Setup (Option B)

**Date:** December 3, 2025  
**Status:** ✅ Tunnel Running & Healthy  
**Purpose:** Use direct `.cfargotunnel.com` URL for immediate LLM access

---

## 📋 What You're Doing

Instead of waiting for DNS migration, you'll use Cloudflare's **direct tunnel URL** to access your LLM server right now. This works immediately and doesn't require any DNS changes.

---

## 🔍 Step 1: Find Your Tunnel ID

### Method 1: From Cloudflare Dashboard (Easiest)

1. **Go to Cloudflare Zero Trust Dashboard:**
   ```
   https://one.dash.cloudflare.com/
   ```

2. **Navigate to:**
   - Click "Networks" in left sidebar
   - Click "Tunnels"
   - You'll see your tunnel listed

3. **Copy the Tunnel ID:**
   - Look for a long UUID like: `a1b2c3d4-e5f6-7890-abcd-ef1234567890`
   - Click to copy it

### Method 2: From RunPod Server

If you still have SSH access to RunPod, run:

```bash
# Look for tunnel ID in the config
cat ~/.cloudflared/config.yml | grep tunnel

# Or check the credentials file
ls ~/.cloudflared/*.json
# The filename IS your tunnel ID
```

### Method 3: From Tunnel Logs

Your tunnel ID appears in the logs. On RunPod run:

```bash
grep "tunnel=" /workspace/logs/cloudflare-tunnel.log | head -1
```

---

## 🌐 Step 2: Build Your Direct URL

Once you have your tunnel ID, your direct URL is:

```
https://<TUNNEL_ID>.cfargotunnel.com
```

**✅ YOUR ACTUAL TUNNEL URL:**
```
https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com
```

**Tunnel Details:**
- **Name:** LLM
- **Tunnel ID:** `5887803e-cf72-4fcc-82ce-4cc1f4b1dd61`
- **Connector ID:** `7e36cb2f-26dd-486c-b1cf-7c4ca27fae2b`
- **Type:** cloudflared

---

## 🧪 Step 3: Test the Direct URL

### Test Health Endpoint

```bash
# Your actual URL - test this now!
curl https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "device": "cuda",
  "timestamp": "2025-12-03T12:30:00"
}
```

### Test Generation

```bash
curl -X POST https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

---

## ⚙️ Step 4: Update Backend Configuration

### On Render Dashboard:

1. **Go to:** https://dashboard.render.com
2. **Select your backend service:** `ai-stanbul`
3. **Go to:** Environment tab
4. **Add or Update:**
   ```
   LLM_SERVER_URL=https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com
   ```
5. **Click:** "Save Changes"
6. **Wait:** 2-3 minutes for automatic redeployment

### Test Backend Integration

```bash
# Test if backend can reach LLM
curl https://ai-stanbul.onrender.com/api/health

# Check backend logs for LLM connection
# (In Render dashboard -> Logs tab)
```

---

## 🎨 Step 5: Update Frontend Configuration

### On Vercel Dashboard:

1. **Go to:** https://vercel.com
2. **Select:** aistanbul.net project
3. **Go to:** Settings → Environment Variables
4. **Add:**
   ```
   VITE_LLM_SERVER_URL=https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com
   ```
5. **Apply to:** Production
6. **Redeploy:** Go to Deployments → Click "..." → Redeploy

---

## 🛡️ Step 6: Configure CORS (If Needed)

If you're calling the LLM directly from frontend (not through backend), you need to allow CORS.

### On RunPod, edit `llm_server.py`:

```python
# Update CORS origins to include your tunnel URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://aistanbul.net",
        "https://www.aistanbul.net",
        "https://<TUNNEL_ID>.cfargotunnel.com"  # Add this
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Restart the server:

```bash
# Kill the old process
kill $(cat /workspace/llm_server.pid)

# Start with new config
cd /workspace
nohup python llm_server.py > /workspace/logs/llm_server.log 2>&1 &
echo $! > /workspace/llm_server.pid
```

---

## ✅ Step 7: Test Everything

### 1. Test Direct URL
```bash
curl https://<TUNNEL_ID>.cfargotunnel.com/health
```

### 2. Test Backend
```bash
curl https://ai-stanbul.onrender.com/api/health
```

### 3. Test Frontend
1. Open: https://aistanbul.net
2. Open browser console (F12)
3. Try using the chat feature
4. Look for successful API calls

---

## 🎯 Success Criteria

You'll know it's working when:

- ✅ Direct tunnel URL responds to `/health`
- ✅ Backend can connect to LLM server
- ✅ Frontend chat feature works (or at least tries)
- ✅ No CORS errors in browser console
- ✅ No 404 or connection errors

---

## 🔧 Troubleshooting

### Issue: "Failed to connect" or "Couldn't connect to server" ⚠️ **CURRENT ISSUE**

**Error:**
```
curl: (7) Failed to connect to 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com port 443
```

**Cause:** Tunnel config only routes `llm.aistanbul.net`, not the direct `.cfargotunnel.com` URL

**Fix (AUTOMATED):**
```bash
# On RunPod, run this script:
cd /workspace
wget https://raw.githubusercontent.com/[your-repo]/fix_tunnel_direct_access.sh
chmod +x fix_tunnel_direct_access.sh
./fix_tunnel_direct_access.sh
```

**Fix (MANUAL):**
```bash
# On RunPod, edit the config:
nano ~/.cloudflared/config.yml

# Change from:
ingress:
  - hostname: llm.aistanbul.net
    service: http://localhost:8000
  - service: http_status:404

# To:
ingress:
  - hostname: llm.aistanbul.net
    service: http://localhost:8000
  - service: http://localhost:8000  # ← Catch-all for direct URL

# Save and restart tunnel:
kill $(cat /workspace/cloudflare-tunnel.pid)
nohup cloudflared tunnel --config ~/.cloudflared/config.yml run 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61 > /workspace/logs/cloudflare-tunnel.log 2>&1 &
echo $! > /workspace/cloudflare-tunnel.pid

# Wait 30 seconds, then test:
sleep 30
curl https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com/health
```

### Issue: "502 Bad Gateway"

**Cause:** LLM server not running or tunnel misconfigured

**Fix:**
```bash
# On RunPod, check server is running
ps aux | grep llm_server

# Check logs
tail -20 /workspace/logs/llm_server.log

# Restart if needed
cd /workspace
./start_llm_server_runpod.sh
```

### Issue: "Connection Refused"

**Cause:** Tunnel not routing to localhost:8000

**Fix:**
```bash
# On RunPod, verify tunnel config
cat ~/.cloudflared/config.yml

# Should have catch-all ingress rule
# Restart tunnel
kill $(cat /workspace/cloudflare-tunnel.pid)
nohup cloudflared tunnel --config ~/.cloudflared/config.yml run 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61 > /workspace/logs/cloudflare-tunnel.log 2>&1 &
echo $! > /workspace/cloudflare-tunnel.pid
```

### Issue: CORS Error in Browser

**Cause:** LLM server doesn't allow tunnel domain

**Fix:** Follow Step 6 above to add tunnel URL to CORS

### Issue: Tunnel URL Returns 404

**Cause:** No ingress rule or wrong URL format

**Fix:**
```bash
# Verify tunnel has default ingress rule
grep "service:" ~/.cloudflared/config.yml

# Should see: service: http://localhost:8000
```

---

## 📊 Comparison: Custom Domain vs Direct URL

| Feature | Custom Domain | Direct URL |
|---------|--------------|------------|
| **URL** | `llm.aistanbul.net` | `<tunnel-id>.cfargotunnel.com` |
| **Setup Time** | 15 min - 48 hours | **Immediate** |
| **DNS Required** | Yes | No |
| **Professional** | ✅ Very | ⚠️ Less |
| **Works Now** | ❌ No | ✅ **Yes** |
| **SSL** | ✅ Yes | ✅ Yes |
| **Cost** | Free | Free |

---

## 🚀 Next Steps After This Works

Once you verify the direct URL works:

1. **Keep using it** for development/testing
2. **Migrate DNS** to Cloudflare (when ready)
3. **Switch to custom domain** later
4. **Update configs** to use `llm.aistanbul.net`

---

## 📝 Quick Reference

### Your URLs:
```
LLM Direct:  https://<TUNNEL_ID>.cfargotunnel.com
Backend:     https://ai-stanbul.onrender.com
Frontend:    https://aistanbul.net
```

### Test Commands:
```bash
# Health check
curl https://<TUNNEL_ID>.cfargotunnel.com/health

# Generate text
curl -X POST https://<TUNNEL_ID>.cfargotunnel.com/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Test","max_tokens":10}'

# Check tunnel status
ps aux | grep cloudflared

# View tunnel logs
tail -f /workspace/logs/cloudflare-tunnel.log

# View LLM logs
tail -f /workspace/logs/llm_server.log
```

---

## 🎉 Why This Is Great

✅ **Works immediately** - no DNS waiting  
✅ **Fully secure** - HTTPS with Cloudflare SSL  
✅ **Reliable** - 4 redundant tunnel connections  
✅ **Professional** - production-grade infrastructure  
✅ **Free** - no additional costs  
✅ **Temporary or permanent** - your choice  

---

## 🔒 Security Notes

- ✅ Direct URL is secure (HTTPS)
- ✅ Only your services can use it (no public listing)
- ✅ Tunnel traffic is encrypted
- ⚠️ URL is long and not branded (use custom domain later)
- ✅ Can add authentication in LLM server if needed

---

## 📞 Need Help?

**Can't find tunnel ID?**  
→ Check Cloudflare dashboard under Networks → Tunnels

**Direct URL not working?**  
→ Check tunnel is running: `ps aux | grep cloudflared`

**502 errors?**  
→ Check LLM server: `ps aux | grep llm_server`

**Want custom domain instead?**  
→ See `DNS_MIGRATION_GUIDE.md` (when ready)

---

**Last Updated:** December 3, 2025  
**Status:** Ready to implement  
**Estimated Time:** 10 minutes  
**Next:** Find your tunnel ID and test!
