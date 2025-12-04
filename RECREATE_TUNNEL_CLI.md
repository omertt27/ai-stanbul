# Recreate Cloudflare Tunnel via CLI (Full Control)

## Problem
Your tunnel was created via the Cloudflare UI, so it ignores your local `~/.cloudflared/config.yml` file. To get full control, recreate it via CLI.

## Solution: Delete and Recreate Tunnel

### Step 1: Delete Existing Tunnel (In Cloudflare Dashboard)
1. Go to: https://one.dash.cloudflare.com/
2. Navigate to **Networks** → **Tunnels**
3. Find your tunnel: `8f83b9a5-d0cf-4a9d-833f-89efc1e9a9a7`
4. Click **Delete**
5. Confirm deletion

### Step 2: SSH to RunPod and Stop Old Tunnel
```bash
ssh root@runpod-server
pkill -f cloudflared
```

### Step 3: Create New Tunnel via CLI
```bash
# Login to Cloudflare (if needed)
cloudflared tunnel login

# Create new tunnel
cloudflared tunnel create istanbul-vllm

# This will output a tunnel ID and create credentials at:
# ~/.cloudflared/<TUNNEL_ID>.json
```

**Save the tunnel ID** that's printed!

### Step 4: Create Config File
```bash
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: istanbul-vllm
credentials-file: /root/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: asdweq123.org
    service: http://localhost:8000
  - hostname: api.asdweq123.org
    service: http://localhost:8000
  - service: http_status:404
EOF
```

**Replace `<TUNNEL_ID>`** with the actual ID from step 3!

### Step 5: Create DNS Records (In Cloudflare Dashboard)
1. Go to: https://dash.cloudflare.com/
2. Select your domain: `asdweq123.org`
3. Go to **DNS** → **Records**
4. Delete old CNAME records for `asdweq123.org` and `api.asdweq123.org`
5. Run this command on RunPod to create new DNS records:

```bash
# For asdweq123.org
cloudflared tunnel route dns istanbul-vllm asdweq123.org

# For api.asdweq123.org
cloudflared tunnel route dns istanbul-vllm api.asdweq123.org
```

This automatically creates CNAME records pointing to your tunnel!

### Step 6: Start Tunnel
```bash
nohup cloudflared tunnel --config ~/.cloudflared/config.yml run istanbul-vllm > /tmp/cloudflared.log 2>&1 &
```

### Step 7: Verify in Logs
```bash
tail -f /tmp/cloudflared.log
```

You should see **BOTH** hostnames:
```json
"ingress":[
  {"hostname":"asdweq123.org","service":"http://localhost:8000"},
  {"hostname":"api.asdweq123.org","service":"http://localhost:8000"},
  {"service":"http_status:404"}
]
```

### Step 8: Test from Mac
```bash
curl -s https://asdweq123.org/health | jq
curl -s https://api.asdweq123.org/health | jq
```

Both should return:
```json
{"status":"ok"}
```

---

## Why This Works
- Tunnels created via **CLI** use your local `~/.cloudflared/config.yml` file
- You have **full control** over ingress rules
- No dashboard configuration needed
- Changes are picked up on tunnel restart

---

## Next Steps
1. Test both endpoints from Mac
2. Update backend `.env` to use working endpoint
3. Test full chat flow from frontend
4. Document final working configuration

---

## Rollback (If Needed)
If something goes wrong, you can recreate the old tunnel:
```bash
cloudflared tunnel create old-tunnel
# Update config.yml with old tunnel ID
# Restart tunnel
```

The old credentials file is still at: `~/.cloudflared/8f83b9a5-d0cf-4a9d-833f-89efc1e9a9a7.json`
