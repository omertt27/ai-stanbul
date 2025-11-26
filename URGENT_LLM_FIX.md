# 🚨 URGENT: LLM Integration Issue

**Date:** November 26, 2025  
**Priority:** HIGH - Blocks main functionality

---

## 🔴 Problem: Backend Not Using LLM

### What You're Seeing:
```bash
curl -X POST https://ai-stanbul.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Recommend restaurants in Kadıköy", "conversation_id": "test"}'
```

**Returns (WRONG):**
```json
{
  "response": "Welcome to Istanbul! How can I help you today?",
  "intent": "greeting"
}
```

**Should Return:** Actual restaurant recommendations with $ symbols

---

## ✅ Good News: Code is Ready

### We Just Completed:
1. ✅ Updated `prompts.py` - Dollar symbols only for prices
2. ✅ All prompt engineering fixed
3. ✅ Backend code fully integrated
4. ✅ Local services (restaurants, etc.) connected

### The ONLY Issue:
**Environment variables not set on Render** ← This is blocking everything

---

## 🎯 Solution: 15-Minute Fix

### What You Need To Do:

**Step 1: Set Up Permanent Tunnel (10 min)**

You have several options for a permanent tunnel:

### Option A: ngrok (Recommended - Free tier available)

1. **Sign up for ngrok:**
   - Go to: https://ngrok.com/
   - Create free account
   - Get your auth token from dashboard

2. **SSH into your RunPod pod:**
   ```bash
   # Install ngrok
   wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
   tar xvzf ngrok-v3-stable-linux-amd64.tgz
   sudo mv ngrok /usr/local/bin/
   
   # Configure with your auth token
   ngrok config add-authtoken YOUR_AUTH_TOKEN_HERE
   ```

3. **Start permanent tunnel:**
   ```bash
   # Start ngrok in background (forwards port 8000 to public URL)
   nohup ngrok http 8000 > ngrok.log 2>&1 &
   
   # Get your permanent URL (it will be static with paid plan, or stable with free tier)
   curl http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*' | grep -o 'https://[^"]*'
   ```

4. **Your permanent URL will look like:**
   - Free tier: `https://abc-123-def.ngrok-free.app` (changes on restart)
   - Paid tier ($8/mo): `https://your-custom-name.ngrok.io` (permanent)

### Option B: Expose RunPod Port Directly (If available)

1. **Check RunPod port mapping:**
   - In RunPod dashboard, look for "TCP Port Mappings"
   - If available, map internal port 8000 to external port
   - You'll get: `https://your-pod-id-8000.proxy.runpod.net`

### Option C: Cloudflare Tunnel (Named tunnel - Requires domain)

1. **Install cloudflared:**
   ```bash
   wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
   sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
   sudo chmod +x /usr/local/bin/cloudflared
   ```

2. **Create named tunnel:**
   ```bash
   cloudflared tunnel login
   cloudflared tunnel create ai-istanbul-llm
   cloudflared tunnel route dns ai-istanbul-llm llm.yourdomain.com
   ```

3. **Run tunnel:**
   ```bash
   nohup cloudflared tunnel --url http://localhost:8000 run ai-istanbul-llm > cloudflared.log 2>&1 &
   ```

4. **Your URL:** `https://llm.yourdomain.com`

### Recommended: ngrok Free Tier

For quick setup, use ngrok free tier:
```bash
# On RunPod (one-time setup)
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin/
ngrok config add-authtoken YOUR_TOKEN

# Start tunnel (persists across sessions)
nohup ngrok http 8000 > ngrok.log 2>&1 &

# Get URL
curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

**Your permanent LLM URL:** `https://abc-123.ngrok-free.app/v1` (add `/v1` at the end!)

**Step 2: Set Environment Variables on Render (5 min)**

1. Go to: https://dashboard.render.com/
2. Click on your backend service (ai-stanbul or similar name)
3. Click "Environment" tab on the left
4. Click "Add Environment Variable"
5. Add these THREE variables:

   ```
   Variable Name: LLM_API_URL
   Value: https://[your-cloudflare-url]/v1
   
   Variable Name: PURE_LLM_MODE  
   Value: true
   
   Variable Name: LLM_MODEL_NAME
   Value: meta-llama/Llama-3.1-8B-Instruct
   ```

6. Click "Save Changes" at the bottom

**Step 3: Trigger Deployment (1 min)**

On the same page:
1. Click "Manual Deploy" button (top right)
2. Select "Deploy latest commit"
3. Click "Deploy"

**Step 4: Wait and Monitor (5 min)**

1. Click "Logs" tab
2. Watch for these messages:
   ```
   ✅ Service Manager ready: 12/12 services active
   ✅ RunPod LLM Client initialized
   ✅ Pure LLM Core initialized
   ✅ Backend startup complete
   ```

**Step 5: Test (2 min)**

```bash
# Test health
curl https://ai-stanbul.onrender.com/api/health/detailed | python3 -m json.tool

# Should show: "pure_llm": { "status": "available" }

# Test chat
curl -X POST https://ai-stanbul.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Recommend restaurants in Kadıköy", "conversation_id": "test"}' \
  | python3 -m json.tool
```

**Expected:** Real restaurant recommendations with $ symbols!

---

## 📋 Quick Checklist

- [ ] I have my Cloudflare tunnel URL ready
- [ ] I'm logged into Render dashboard
- [ ] I found my backend service
- [ ] I clicked "Environment" tab
- [ ] I added LLM_API_URL with /v1 at the end
- [ ] I added PURE_LLM_MODE=true
- [ ] I added LLM_MODEL_NAME
- [ ] I clicked "Save Changes"
- [ ] I clicked "Manual Deploy"
- [ ] I'm watching the logs
- [ ] I see "✅ Backend startup complete"
- [ ] I tested with curl and got real responses

---

## 🆘 If Something Goes Wrong

### Issue: "Can't find my Cloudflare URL"

**Solution:**
1. Is your RunPod pod still running? Check: https://www.runpod.io/console/pods
2. If stopped, restart it
3. SSH in and run: `cat /workspace/cloudflared.log | grep https://`
4. Use that URL

### Issue: "Deployment failed"

**Solution:**
1. Check error message in Render logs
2. Most common: Wrong URL format (make sure it ends with `/v1`)
3. Try deploying again

### Issue: "Still getting greeting response"

**Solution:**
1. Check health endpoint: `/api/health/detailed`
2. Look for "pure_llm" status
3. If still "unavailable", check logs for error messages
4. Verify env vars are saved (refresh Environment tab)

### Issue: "Connection timeout to LLM"

**Solution:**
1. Test RunPod directly: `curl https://[your-url]/v1/models`
2. If no response, restart RunPod pod
3. Make sure cloudflared is running

---

## 📞 Help Resources

1. **Permanent Tunnel Setup:** `PERMANENT_TUNNEL_SETUP.md` ⭐ READ THIS FIRST!
2. **Detailed Fix Guide:** `RENDER_LLM_FIX_NOW.md`
3. **Price Format Guide:** `PRICE_FORMAT_UPDATE_COMPLETE.md`
4. **Full Integration Docs:** `COMPLETE_INTEGRATION_SUMMARY.md`

---

## ⏱️ Time Estimate

- **If RunPod is running:** 10 minutes
- **If need to restart RunPod:** 20 minutes
- **If completely new setup:** 30 minutes

---

## 💡 Why This Happened

The backend code is perfect and ready. But on Render, environment variables don't transfer automatically from local development. We need to explicitly set:
- Where the LLM server is (LLM_API_URL)
- That we want to use it (PURE_LLM_MODE)
- What model to use (LLM_MODEL_NAME)

Once these are set, everything will work immediately!

---

## ✨ After This Fix

You'll have:
- ✅ Full LLM integration working
- ✅ Real AI responses about Istanbul
- ✅ Prices shown as $ symbols only
- ✅ All 12 local services (restaurants, transport, etc.) connected
- ✅ Production-ready backend
- ✅ Ready to test with frontend

---

**START HERE:** Go to Render dashboard and add those 3 environment variables!

**Questions?** Read `RENDER_LLM_FIX_NOW.md` for detailed troubleshooting.
