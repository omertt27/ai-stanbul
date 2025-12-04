# 🔧 Fix Cloudflare Tunnel - Missing Hostname

## ❌ Problem Identified

Your `~/.cloudflared/config.yml` is missing the **hostname** configuration:

**Current (Broken):**
```yaml
tunnel: 3c9f3076-300f-4a61-b923-cf7be81e2919
credentials-file: /root/.cloudflared/3c9f3076-300f-4a61-b923-cf7be81e2919.json

ingress:
  - service: http://localhost:8000
```

**Should be:**
```yaml
tunnel: 3c9f3076-300f-4a61-b923-cf7be81e2919
credentials-file: /root/.cloudflared/3c9f3076-300f-4a61-b923-cf7be81e2919.json

ingress:
  - hostname: asdweq123.org
    service: http://localhost:8000
  - service: http_status:404
```

---

## 🚀 Quick Fix - Run on RunPod Terminal

### Step 1: Update the Config File

```bash
# Backup current config
cp ~/.cloudflared/config.yml ~/.cloudflared/config.yml.backup

# Create new config with hostname
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: 3c9f3076-300f-4a61-b923-cf7be81e2919
credentials-file: /root/.cloudflared/3c9f3076-300f-4a61-b923-cf7be81e2919.json

ingress:
  - hostname: asdweq123.org
    service: http://localhost:8000
  - service: http_status:404
EOF

# Verify the new config
cat ~/.cloudflared/config.yml
```

### Step 2: Restart Cloudflare Tunnel

```bash
# Kill the current tunnel
pkill cloudflared

# Wait a moment
sleep 2

# Start with the new config
nohup cloudflared tunnel run 3c9f3076-300f-4a61-b923-cf7be81e2919 > /workspace/cloudflared.log 2>&1 &

# Check if it started
sleep 3
ps aux | grep cloudflared | grep -v grep

# View logs to check for errors
tail -50 /workspace/cloudflared.log
```

### Step 3: Configure DNS in Cloudflare Dashboard

**Option A: Automatic (Recommended)**
```bash
# Let cloudflared set up DNS automatically
cloudflared tunnel route dns 3c9f3076-300f-4a61-b923-cf7be81e2919 asdweq123.org
```

**Option B: Manual (via Web Dashboard)**

1. Go to: https://dash.cloudflare.com
2. Select domain: `asdweq123.org`
3. Go to **DNS** → **Records**
4. Click **Add record**
5. Configure:
   - **Type:** CNAME
   - **Name:** `@` (for root domain) or `vllm` (for subdomain)
   - **Target:** `3c9f3076-300f-4a61-b923-cf7be81e2919.cfargotunnel.com`
   - **Proxy status:** Proxied (orange cloud) ✅
   - **TTL:** Auto
6. Click **Save**

### Step 4: Test the Tunnel

```bash
# Wait for DNS to propagate (30-60 seconds)
sleep 60

# Test from RunPod
curl -s -w "\nHTTP Code: %{http_code}\n" https://asdweq123.org/health --connect-timeout 10

# If it works, test a completion
curl -X POST https://asdweq123.org/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", "prompt": "Hello", "max_tokens": 10}' \
  --connect-timeout 30
```

---

## 🎯 All-in-One Fix Script

Copy and paste this entire block into your RunPod terminal:

```bash
#!/bin/bash
echo "🔧 Fixing Cloudflare Tunnel Configuration"
echo "=========================================="
echo ""

TUNNEL_ID="3c9f3076-300f-4a61-b923-cf7be81e2919"
TUNNEL_DOMAIN="asdweq123.org"

# 1. Backup and update config
echo "1️⃣ Updating config.yml..."
cp ~/.cloudflared/config.yml ~/.cloudflared/config.yml.backup 2>/dev/null || true

cat > ~/.cloudflared/config.yml << EOF
tunnel: ${TUNNEL_ID}
credentials-file: /root/.cloudflared/${TUNNEL_ID}.json

ingress:
  - hostname: ${TUNNEL_DOMAIN}
    service: http://localhost:8000
  - service: http_status:404
EOF

echo "✅ Config updated"
cat ~/.cloudflared/config.yml
echo ""

# 2. Restart tunnel
echo "2️⃣ Restarting cloudflared..."
pkill cloudflared 2>/dev/null || true
sleep 2

nohup cloudflared tunnel run ${TUNNEL_ID} > /workspace/cloudflared.log 2>&1 &
sleep 3

if ps aux | grep cloudflared | grep -v grep > /dev/null; then
    echo "✅ Cloudflared is running"
    ps aux | grep cloudflared | grep -v grep | head -1
else
    echo "❌ Failed to start cloudflared"
    echo "Logs:"
    tail -20 /workspace/cloudflared.log
    exit 1
fi
echo ""

# 3. Setup DNS
echo "3️⃣ Setting up DNS..."
cloudflared tunnel route dns ${TUNNEL_ID} ${TUNNEL_DOMAIN}
echo ""

# 4. Wait for DNS propagation
echo "4️⃣ Waiting for DNS propagation (60 seconds)..."
sleep 60
echo ""

# 5. Test tunnel
echo "5️⃣ Testing tunnel..."
echo "Testing: https://${TUNNEL_DOMAIN}/health"
RESULT=$(curl -s -w "\nHTTP_CODE:%{http_code}" https://${TUNNEL_DOMAIN}/health --connect-timeout 15 2>&1)
HTTP_CODE=$(echo "$RESULT" | grep "HTTP_CODE:" | cut -d: -f2)

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ SUCCESS! Tunnel is working!"
    echo ""
    echo "Response:"
    echo "$RESULT" | grep -v "HTTP_CODE:"
    echo ""
    echo "🎉 Your vLLM is now accessible at: https://${TUNNEL_DOMAIN}"
    echo ""
    echo "Next steps:"
    echo "1. Update .env: LLM_API_URL=https://${TUNNEL_DOMAIN}"
    echo "2. Update Render.com env vars"
    echo "3. Test chat from frontend"
else
    echo "⚠️  Tunnel test failed (HTTP Code: $HTTP_CODE)"
    echo "Response: $RESULT"
    echo ""
    echo "Check logs:"
    echo "  tail -50 /workspace/cloudflared.log"
    echo ""
    echo "Check DNS:"
    echo "  dig ${TUNNEL_DOMAIN}"
    echo "  nslookup ${TUNNEL_DOMAIN}"
fi
```

---

## 📋 What Each Step Does

1. **Update config.yml** - Adds the missing hostname
2. **Restart tunnel** - Applies the new configuration
3. **Setup DNS** - Creates CNAME record in Cloudflare
4. **Wait** - Allows DNS to propagate
5. **Test** - Verifies the tunnel is accessible

---

## 🔍 Troubleshooting

### If DNS setup fails with "route dns" command:

You may need to set it up manually in Cloudflare dashboard:

1. Go to: https://dash.cloudflare.com
2. Select `asdweq123.org`
3. DNS → Add Record:
   - Type: CNAME
   - Name: @ (or subdomain like `vllm`)
   - Content: `3c9f3076-300f-4a61-b923-cf7be81e2919.cfargotunnel.com`
   - Proxy: ON (orange cloud)

### Check Tunnel Logs

```bash
tail -100 /workspace/cloudflared.log
```

### Verify DNS Resolution

```bash
# From RunPod
nslookup asdweq123.org

# Or
dig asdweq123.org
```

### Test from Your Mac (After Fix)

```bash
# From your Mac terminal
curl https://asdweq123.org/health
```

---

## ✅ Success Criteria

Once fixed, you should see:

```bash
$ curl https://asdweq123.org/health
{"status":"healthy","model_loaded":true,"model_name":"meta-llama/Meta-Llama-3.1-8B-Instruct",...}
```

Then your tunnel is ready for production! 🚀
