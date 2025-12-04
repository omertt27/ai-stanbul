# 🎯 Configure Cloudflare Tunnel Service - Step by Step

## You're almost there! Just need to configure the service.

### Current Location
You're viewing: **Connector diagnostics** for tunnel "llama"

### What You Need to Do

1. **Look for tabs at the top** of the tunnel configuration page:
   - Overview
   - **Public Hostname** (or **Hostname routes**) ← Click this!
   - Private Networks
   - Connector diagnostics (you're here now)

2. **Click on "Public Hostname" or "Hostname routes" tab**

3. **You should see your hostnames listed:**
   - `asdweq123.org`
   - Possibly `api.asdweq123.org`

4. **Click Edit (pencil icon) next to `asdweq123.org`**

5. **Configure the service section:**
   
   Look for a section that says "Service" or "Backend" with these fields:
   
   | Field | Value to Enter |
   |-------|----------------|
   | **Type** | Select: `HTTP` |
   | **URL** | Type: `localhost:8000` |

   **Important:** Do NOT use `https://`, just `localhost:8000` or `http://localhost:8000`

6. **Save the hostname**

7. **Repeat for `api.asdweq123.org`** if it exists

8. **Test from your Mac:**
   ```bash
   curl -s https://asdweq123.org/health
   ```

---

## Visual Guide

The page structure should look like:

```
┌─────────────────────────────────────────┐
│ Tunnel: llama                           │
├─────────────────────────────────────────┤
│ [Overview] [Public Hostname] [Private Networks] [Diagnostics] │
│                    ↑                     │
│             Click this tab!              │
└─────────────────────────────────────────┘

Then you'll see:

┌─────────────────────────────────────────┐
│ Public Hostname                          │
│ [Add a public hostname]                  │
├─────────────────────────────────────────┤
│ Hostname: asdweq123.org         [Edit]   │
│ Service: (need to configure)             │
│                                          │
│ Hostname: api.asdweq123.org     [Edit]   │
│ Service: (need to configure)             │
└─────────────────────────────────────────┘
```

---

## If You Don't See "Public Hostname" Tab

The tunnel configuration might use different wording. Look for:
- **Hostname routes**
- **Public hostnames**
- **Configure**

Or try going back to the tunnels list:
1. Go to: https://one.dash.cloudflare.com
2. Networks & Security → Tunnels
3. Click "Configure" button next to "llama" tunnel
4. Look for tabs/sections to configure hostnames

---

## Quick Check - Are Services Configured?

If you see your hostnames (`asdweq123.org` and `api.asdweq123.org`) but they don't have services configured, that's the problem!

Each hostname needs:
- ✅ Hostname: asdweq123.org
- ❌ Service: **Not configured** ← This is why you get 503 error!

Should be:
- ✅ Hostname: asdweq123.org  
- ✅ Service: HTTP → localhost:8000

---

## Once Configured Correctly

Test from your Mac:
```bash
# Should return healthy status
curl -s https://asdweq123.org/health

# Expected output:
# {"status":"healthy","model_loaded":true,"model_name":"meta-llama/Meta-Llama-3.1-8B-Instruct",...}
```

---

Let me know what you see when you click on the "Public Hostname" or "Hostname routes" section! 🚀
