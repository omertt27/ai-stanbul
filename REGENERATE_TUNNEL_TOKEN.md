# 🔄 Regenerate Cloudflare Tunnel Token

**Date:** December 3, 2025  
**Issue:** Tunnel showing "Unauthorized: Failed to get tunnel"  
**Cause:** Deleting routes invalidated the token  
**Solution:** Get a fresh token from Cloudflare dashboard

---

## 🎯 Why This Happened

When you deleted the hostname routes in Cloudflare dashboard, it may have also removed or reset the tunnel configuration, causing the old token to become unauthorized.

---

## ✅ Step-by-Step Fix

### Step 1: Go to Cloudflare Dashboard

1. **Open:** https://one.dash.cloudflare.com/
2. **Navigate to:** Networks → Tunnels
3. **Find your tunnel:** `LLM` (ID: 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61)

### Step 2: Configure the Tunnel

You have two options:

#### Option A: Add a Public Hostname (Simplest)

1. **Click** on your tunnel name `LLM`
2. **Go to** "Public Hostname" tab
3. **Click** "Add a public hostname"
4. **Fill in:**
   - **Subdomain:** `*` (catch-all)
   - **Domain:** Leave empty or select your domain
   - **Service Type:** `HTTP`
   - **URL:** `localhost:8000`
5. **Click** "Save hostname"

This creates a catch-all rule that routes **everything** (including `.cfargotunnel.com`) to your LLM server.

#### Option B: Get Install Command with Fresh Token

1. **Click** on your tunnel name `LLM`
2. **Look for** "Install and run a connector" section
3. **Select:** Operating System → Linux
4. **Copy** the command that looks like:
   ```bash
   cloudflared tunnel run --token eyJ...NEW_TOKEN_HERE...
   ```

### Step 3: Use the New Token on RunPod

Once you have the fresh token/command:

```bash
# Stop old tunnel
pkill -f cloudflared
sleep 3

# Start with NEW token (replace with your actual new token)
nohup cloudflared tunnel run --token <YOUR_NEW_TOKEN> > /workspace/logs/cloudflare-tunnel.log 2>&1 &

# Save PID
echo $! > /workspace/cloudflare-tunnel.pid

# Wait for initialization
sleep 30

# Check logs
tail -20 /workspace/logs/cloudflare-tunnel.log

# Test
curl https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com/health
```

---

## 🎯 Alternative: Create a New Tunnel

If the above doesn't work, you can create a brand new tunnel:

### Step 1: Delete Old Tunnel (Optional)
1. In Cloudflare dashboard → Networks → Tunnels
2. Click "..." menu on your `LLM` tunnel
3. Click "Delete tunnel"

### Step 2: Create New Tunnel
1. Click "Create a tunnel"
2. **Name:** `LLM-Direct`
3. **Connector type:** Cloudflared
4. Click "Save tunnel"

### Step 3: Configure Ingress
1. **Skip** the public hostname section (or add catch-all)
2. Go to next step
3. Copy the installation command with the token

### Step 4: Run on RunPod
```bash
# Use the command from dashboard
cloudflared tunnel run --token <NEW_TOKEN>
```

---

## 🔍 What to Look For

### ✅ Success Indicators:
```
INF Registered tunnel connection connIndex=0
INF Registered tunnel connection connIndex=1
INF Registered tunnel connection connIndex=2
INF Registered tunnel connection connIndex=3
```

### ❌ Error Indicators:
```
ERR Register tunnel error from server side error="Unauthorized"
ERR Failed to get tunnel
```

---

## 📋 Quick Reference

### Current Tunnel Info:
- **Tunnel Name:** LLM
- **Tunnel ID:** 5887803e-cf72-4fcc-82ce-4cc1f4b1dd61
- **Direct URL:** https://5887803e-cf72-4fcc-82ce-4cc1f4b1dd61.cfargotunnel.com
- **Status:** ⚠️ Unauthorized (needs fresh token)

### What You Need:
1. Go to Cloudflare dashboard
2. Find your tunnel
3. Get fresh token OR configure public hostname
4. Restart tunnel on RunPod with new token

---

## 🎯 Recommended Quick Fix

**The fastest solution:**

1. **Go to:** https://one.dash.cloudflare.com/ → Networks → Tunnels
2. **Click:** Your `LLM` tunnel
3. **Click:** "Public Hostname" tab
4. **Click:** "Add a public hostname"
5. **Configure:**
   - Service: HTTP
   - URL: localhost:8000
   - Subdomain: * (or leave empty for catch-all)
6. **Save**
7. **Go to:** "Install connector" or similar section
8. **Copy** the install command with token
9. **Run** on RunPod:
   ```bash
   pkill -f cloudflared
   # Paste the command from dashboard
   nohup cloudflared tunnel run --token <NEW_TOKEN> > /workspace/logs/cloudflare-tunnel.log 2>&1 &
   echo $! > /workspace/cloudflare-tunnel.pid
   ```

---

## 🛠️ Troubleshooting

### Can't Find Install Command?

Look for sections named:
- "Install and run a connector"
- "Install connector"
- "Configure"
- Click the "?" or "Install" button

### Still Getting Unauthorized?

The tunnel might be deleted. Create a new one:
1. Delete old tunnel in dashboard
2. Create new tunnel
3. Get new token
4. Start fresh on RunPod

### Want to Keep Same Tunnel ID?

You can't - once deleted or invalidated, you need a new tunnel. But the new tunnel will work the same way, just with a different URL.

---

## 📝 After You Get It Working

Once the tunnel is working again:

1. ✅ **Save the new token** somewhere safe
2. ✅ **Update your documentation** with the new URL
3. ✅ **Test the new direct URL**
4. ✅ **Update backend/frontend** with new URL if the tunnel ID changed

---

## 🎉 Why This Will Work

- Fresh token = fresh authorization
- Cloudflare will recognize the connector
- Tunnel will establish connections
- Direct URL will work

---

**Last Updated:** December 3, 2025  
**Status:** Waiting for fresh token from Cloudflare dashboard  
**Next:** Get new token → Restart tunnel → Test  
**Time Required:** 5 minutes
