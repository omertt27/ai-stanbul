# 🏗️ AI Istanbul - Complete Architecture

## Current Infrastructure

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRODUCTION ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────┘

User Browser
    ↓
Vercel (Frontend)
https://your-app.vercel.app
    ↓
Render.com (Backend API)
https://your-backend.onrender.com
    ↓
Cloudflare Tunnel
https://api.asdweq123.org  ← DNS CONFIGURED
    ↓
RunPod (vLLM Server)
http://localhost:8000  ← RUNNING ✅
```

---

## 🔍 Current Status Analysis

Based on your RunPod logs, I found the issue:

### ❌ **Problem: Hostname Mismatch**

**Config file (`~/.cloudflared/config.yml`):**
```yaml
ingress:
  - hostname: asdweq123.org    # ← You want this
    service: http://localhost:8000
```

**What Cloudflare is actually using (from logs):**
```json
"hostname":"api.asdweq123.org"    # ← Cloudflare is using this
```

### 🎯 **Root Cause**

The DNS in Cloudflare was set up for `api.asdweq123.org`, but your config file says `asdweq123.org`. 

**When I tested:**
- ❌ `asdweq123.org` → No DNS record
- ✅ `api.asdweq123.org` → DNS resolves (104.21.23.64, 172.67.209.119)
- ❌ But returns 503 error (tunnel not routing correctly)

---

## 🚀 Solution: Fix the Tunnel Configuration

You have **2 options**:

### Option A: Use Root Domain (asdweq123.org) - RECOMMENDED

Run this on **RunPod terminal**:

```bash
#!/bin/bash
TUNNEL_ID="3c9f3076-300f-4a61-b923-cf7be81e2919"

echo "🔧 Fixing Cloudflare Tunnel for asdweq123.org"
echo ""

# 1. Update Cloudflare DNS to use root domain
echo "1️⃣ Setting up DNS for root domain..."
cloudflared tunnel route dns ${TUNNEL_ID} asdweq123.org

# 2. Verify config
echo ""
echo "2️⃣ Current config:"
cat ~/.cloudflared/config.yml

# 3. Restart tunnel (config is already correct)
echo ""
echo "3️⃣ Restarting tunnel..."
pkill cloudflared
sleep 2
nohup cloudflared tunnel run ${TUNNEL_ID} > /workspace/cloudflared.log 2>&1 &

# 4. Wait for startup
sleep 5

# 5. Check logs
echo ""
echo "4️⃣ Checking logs..."
tail -10 /workspace/cloudflared.log | grep -i hostname

# 6. Test
echo ""
echo "5️⃣ Testing (waiting 30s for DNS)..."
sleep 30
curl -s -w "\nHTTP Code: %{http_code}\n" https://asdweq123.org/health --connect-timeout 10
```

### Option B: Keep api.asdweq123.org (Current DNS)

Run this on **RunPod terminal**:

```bash
#!/bin/bash
TUNNEL_ID="3c9f3076-300f-4a61-b923-cf7be81e2919"

echo "🔧 Fixing Cloudflare Tunnel for api.asdweq123.org"
echo ""

# 1. Update config to match DNS
cat > ~/.cloudflared/config.yml << EOF
tunnel: ${TUNNEL_ID}
credentials-file: /root/.cloudflared/${TUNNEL_ID}.json

ingress:
  - hostname: api.asdweq123.org
    service: http://localhost:8000
  - service: http_status:404
EOF

echo "1️⃣ Updated config:"
cat ~/.cloudflared/config.yml

# 2. Restart tunnel
echo ""
echo "2️⃣ Restarting tunnel..."
pkill cloudflared
sleep 2
nohup cloudflared tunnel run ${TUNNEL_ID} > /workspace/cloudflared.log 2>&1 &

# 3. Wait and check
sleep 5
echo ""
echo "3️⃣ Checking logs..."
tail -10 /workspace/cloudflared.log

# 4. Test
echo ""
echo "4️⃣ Testing..."
sleep 10
curl -s -w "\nHTTP Code: %{http_code}\n" https://api.asdweq123.org/health --connect-timeout 10
```

---

## 📋 After Tunnel is Working

### Step 1: Update Local Development

```bash
# On your Mac
cd /Users/omer/Desktop/ai-stanbul

# Edit .env
# Change:
LLM_API_URL=https://asdweq123.org
# Or if using subdomain:
LLM_API_URL=https://api.asdweq123.org
```

### Step 2: Update Render.com Backend

1. Go to: https://dashboard.render.com
2. Select your backend service
3. Go to **Environment** tab
4. Add/Update variables:
   ```
   LLM_API_URL=https://asdweq123.org
   LLM_API_URL_FALLBACK=https://4r1su4zfuok0s7-8000.proxy.runpod.net
   ```
5. Click **Save Changes** (will trigger redeploy)

### Step 3: Verify Frontend (Vercel)

1. Go to: https://vercel.com/dashboard
2. Select your frontend project
3. Check **Environment Variables**
4. Ensure `NEXT_PUBLIC_API_URL` or similar points to your Render backend:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend.onrender.com
   ```

---

## 🧪 Testing Checklist

After fixing the tunnel, test each layer:

### 1. Test vLLM Directly (from RunPod)
```bash
curl http://localhost:8000/health
```
**Expected:** `{"status":"healthy",...}`

### 2. Test Tunnel (from your Mac)
```bash
curl https://asdweq123.org/health
# or
curl https://api.asdweq123.org/health
```
**Expected:** `{"status":"healthy",...}`

### 3. Test Backend (from your Mac)
```bash
curl https://your-backend.onrender.com/health
curl https://your-backend.onrender.com/api/debug/llm-status
```
**Expected:** Backend health and LLM connection status

### 4. Test Frontend Chat
- Open: `https://your-app.vercel.app`
- Type a message in chat
- Should get LLM response (not fallback error)

---

## 🎯 Decision Time

**Which domain do you want to use?**

### Option A: `asdweq123.org` (root domain)
✅ Cleaner URL  
✅ Config already correct  
❓ Need to set up DNS  

### Option B: `api.asdweq123.org` (subdomain)
✅ DNS already working  
❓ Need to update config file  

**My recommendation: Option A (asdweq123.org)** - cleaner and your config is already set for it.

---

## 📝 Summary of What We Found

| Component | Status | Details |
|-----------|--------|---------|
| **vLLM** | ✅ Running | Port 8000, healthy, responding |
| **Cloudflared** | ✅ Running | Process active, 4 connections |
| **Config File** | ✅ Correct | Hostname: `asdweq123.org` |
| **DNS** | ❌ **Mismatch** | Only `api.asdweq123.org` has DNS |
| **Tunnel Route** | ❌ **Wrong** | Routes to `api.asdweq123.org` not root |

**Fix needed:** Update DNS to route `asdweq123.org` (root domain) to tunnel

---

## 🚀 Quick Fix (Run on RunPod)

```bash
# Route root domain to tunnel
cloudflared tunnel route dns 3c9f3076-300f-4a61-b923-cf7be81e2919 asdweq123.org

# Wait for DNS propagation
sleep 60

# Test
curl -s https://asdweq123.org/health
```

**Then share the output and we'll update your .env and Render!** 🎯
