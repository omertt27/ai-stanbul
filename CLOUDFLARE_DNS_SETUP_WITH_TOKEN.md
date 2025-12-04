# 🔐 Setup Cloudflare DNS with API Token

## Option 1: Manual DNS Setup (Easiest) ⭐

Go to Cloudflare Dashboard and add DNS manually:

1. **Go to:** https://dash.cloudflare.com
2. **Select domain:** `asdweq123.org`
3. **Go to:** DNS → Records
4. **Click:** Add record
5. **Configure:**
   - **Type:** CNAME
   - **Name:** `@` (for root domain asdweq123.org)
   - **Target:** `3c9f3076-300f-4a61-b923-cf7be81e2919.cfargotunnel.com`
   - **Proxy status:** Proxied (orange cloud ☁️)
   - **TTL:** Auto
6. **Click:** Save

**Then wait 1-2 minutes and test:**
```bash
curl https://asdweq123.org/health
```

---

## Option 2: Use API Token (From Your Mac)

If you want to use the API token, run these commands **on your Mac**:

### Step 1: Set Your API Token
```bash
# Replace YOUR_TOKEN with your actual Cloudflare API token
export CF_API_TOKEN="YOUR_CLOUDFLARE_API_TOKEN"

# Replace with your Zone ID (found in Cloudflare dashboard overview)
export ZONE_ID="YOUR_ZONE_ID"
```

### Step 2: Add DNS Record via API
```bash
# Add CNAME record for root domain
curl -X POST "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records" \
  -H "Authorization: Bearer ${CF_API_TOKEN}" \
  -H "Content-Type: application/json" \
  --data '{
    "type": "CNAME",
    "name": "@",
    "content": "3c9f3076-300f-4a61-b923-cf7be81e2919.cfargotunnel.com",
    "ttl": 1,
    "proxied": true
  }'
```

### Step 3: Verify DNS Record Was Created
```bash
# List DNS records
curl -X GET "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records?type=CNAME&name=asdweq123.org" \
  -H "Authorization: Bearer ${CF_API_TOKEN}" \
  -H "Content-Type: application/json"
```

### Step 4: Test
```bash
# Wait for DNS propagation (30-60 seconds)
sleep 60

# Test the tunnel
curl https://asdweq123.org/health
```

---

## Option 3: Use API Token on RunPod

If you prefer to do it from RunPod, run these commands **on RunPod terminal**:

```bash
# Set your credentials
export CF_API_TOKEN="YOUR_CLOUDFLARE_API_TOKEN"
export ZONE_ID="YOUR_ZONE_ID"
export TUNNEL_ID="3c9f3076-300f-4a61-b923-cf7be81e2919"

# Add DNS record
curl -X POST "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records" \
  -H "Authorization: Bearer ${CF_API_TOKEN}" \
  -H "Content-Type: application/json" \
  --data "{
    \"type\": \"CNAME\",
    \"name\": \"@\",
    \"content\": \"${TUNNEL_ID}.cfargotunnel.com\",
    \"ttl\": 1,
    \"proxied\": true
  }"

# Wait and test
sleep 60
curl -s https://asdweq123.org/health
```

---

## Finding Your Zone ID

1. Go to: https://dash.cloudflare.com
2. Select domain: `asdweq123.org`
3. Scroll down on the right side
4. Look for **Zone ID** (it looks like: `1234567890abcdef1234567890abcdef`)
5. Copy it

---

## What to Do After DNS is Set Up

Once DNS is working (you get HTTP 200 from `https://asdweq123.org/health`):

### 1. Update Local .env
```bash
cd /Users/omer/Desktop/ai-stanbul
# Edit .env file
LLM_API_URL=https://asdweq123.org
```

### 2. Update Render.com
- Go to: https://dashboard.render.com
- Select your backend service
- Environment tab
- Update: `LLM_API_URL=https://asdweq123.org`
- Save (triggers redeploy)

### 3. Test End-to-End
```bash
# Test from your Mac
curl https://asdweq123.org/health
curl https://your-backend.onrender.com/api/debug/llm-status

# Then test chat from frontend
open https://your-app.vercel.app
```

---

## Quick Commands Reference

### Test vLLM on RunPod
```bash
curl http://localhost:8000/health
```

### Test Tunnel
```bash
curl https://asdweq123.org/health
```

### Test Backend
```bash
curl https://your-backend.onrender.com/health
```

### Check DNS Resolution
```bash
nslookup asdweq123.org
dig asdweq123.org
```

---

## Troubleshooting

### If DNS doesn't work after adding:

1. **Check if record was created:**
   - Go to Cloudflare dashboard
   - DNS → Records
   - Look for CNAME pointing to `3c9f3076-300f-4a61-b923-cf7be81e2919.cfargotunnel.com`

2. **Ensure proxy is ON:**
   - The cloud icon should be orange ☁️ (not gray)

3. **Wait longer:**
   - DNS can take up to 5 minutes to propagate

4. **Check tunnel logs:**
   ```bash
   # On RunPod
   tail -30 /workspace/cloudflared.log | grep -i error
   ```

---

## My Recommendation

**Use Option 1 (Manual Dashboard)** - It's the quickest and most reliable:
1. Add CNAME record in dashboard (2 minutes)
2. Wait 1-2 minutes
3. Test: `curl https://asdweq123.org/health`
4. Done! ✅

Then let me know when it works and I'll help update your .env and Render! 🚀
