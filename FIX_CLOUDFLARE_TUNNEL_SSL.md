# 🚨 Cloudflare Tunnel SSL Issue - Complete Fix

## Problem Identified

- ✅ DNS resolves: `api.asdweq123.org` → 104.21.23.64, 172.67.209.119
- ✅ Tunnel is running on RunPod
- ✅ vLLM is healthy on localhost:8000
- ❌ **Connection reset by peer during SSL handshake**

This means the tunnel configuration in Cloudflare Zero Trust dashboard needs to be checked/fixed.

---

## Solution: Configure Tunnel in Cloudflare Zero Trust

### Step 1: Access Cloudflare Zero Trust

1. Go to: https://one.dash.cloudflare.com
2. Select your account
3. Go to: **Access** → **Tunnels**

### Step 2: Find Your Tunnel

Look for tunnel ID: `3c9f3076-300f-4a61-b923-cf7be81e2919`

Or search for: `ai-istanbul` or similar name

### Step 3: Configure Public Hostname

1. Click on your tunnel
2. Click **Configure**
3. Go to **Public Hostname** tab
4. Click **Add a public hostname**

### Step 4: Add Hostname Configuration

**Fill in these exact values:**

| Field | Value |
|-------|-------|
| **Subdomain** | `api` |
| **Domain** | `asdweq123.org` |
| **Path** | (leave empty) |
| **Type** | `HTTP` |
| **URL** | `localhost:8000` |

### Step 5: Save

Click **Save hostname**

### Step 6: Wait and Test

Wait 1-2 minutes, then test from your Mac:

```bash
curl -s https://api.asdweq123.org/health
```

---

## Alternative: Use Cloudflare Warp Settings

If the above doesn't work, check these settings in the tunnel configuration:

### Check TLS Settings:

1. In tunnel configuration
2. Look for **TLS/SSL** settings
3. Ensure:
   - TLS is enabled
   - Origin Server Name: `localhost` or leave empty
   - No-TLS-Verify: **OFF** (should be secure)

---

## Or: Try HTTP Instead of HTTPS

If nothing else works, test with HTTP temporarily:

```bash
# On your Mac
curl -s http://api.asdweq123.org/health
```

If HTTP works but HTTPS doesn't, it's definitely a Cloudflare SSL configuration issue.

---

## Quick Diagnostic Commands

### On Your Mac:
```bash
# Check DNS
dig api.asdweq123.org

# Test with different protocols
curl -v http://api.asdweq123.org/health 2>&1 | grep "HTTP"
curl -v https://api.asdweq123.org/health 2>&1 | grep "HTTP"

# Check SSL certificate
openssl s_client -connect api.asdweq123.org:443 -servername api.asdweq123.org < /dev/null 2>/dev/null | grep "subject="
```

### On RunPod:
```bash
# Check if vLLM is responding
curl -s http://localhost:8000/health

# Check tunnel logs for errors
tail -50 /workspace/cloudflared.log | grep -i "error\|fail"

# Check if tunnel is receiving requests
tail -f /workspace/cloudflared.log
# Then from Mac: curl https://api.asdweq123.org/health
# Watch for incoming requests in logs
```

---

## Most Likely Solution

The tunnel was probably set up initially with a quick command and doesn't have a proper public hostname configured in Cloudflare Zero Trust dashboard. 

**Go to Zero Trust dashboard and add the public hostname configuration** as described in Steps 1-6 above.

---

## If Still Not Working

Try deleting the tunnel config in Cloudflare dashboard and recreating it:

1. Zero Trust → Tunnels
2. Find your tunnel
3. Configure → Public Hostnames
4. Delete any existing hostname configs
5. Add new hostname:
   - Subdomain: `api`
   - Domain: `asdweq123.org` 
   - Service: `http://localhost:8000`
6. Save

Then restart tunnel on RunPod:
```bash
pkill cloudflared
sleep 2
cloudflared tunnel run 3c9f3076-300f-4a61-b923-cf7be81e2919 &
```

---

## Success Indicators

Once working, you should see:

```bash
$ curl -s https://api.asdweq123.org/health
{"status":"healthy","model_loaded":true,"model_name":"meta-llama/Meta-Llama-3.1-8B-Instruct",...}
```

Then we can proceed to update your `.env` and Render.com! 🚀
