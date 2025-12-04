# Fix Cloudflare Tunnel in Dashboard

## Problem
Your tunnel was created via the Cloudflare UI, so it ignores your local `~/.cloudflared/config.yml` file. The dashboard only shows `api.asdweq123.org`, not `asdweq123.org`.

## Solution: Add Both Hostnames in Cloudflare Dashboard

### Step 1: Go to Cloudflare Zero Trust Dashboard
1. Go to: https://one.dash.cloudflare.com/
2. Navigate to **Networks** → **Tunnels**
3. Find your tunnel: `8f83b9a5-d0cf-4a9d-833f-89efc1e9a9a7`
4. Click **Edit** (or the **3 dots** → **Configure**)

### Step 2: Add Public Hostnames
1. Go to the **Public Hostname** tab
2. You should see **one entry** for `api.asdweq123.org`
3. Click **Add a public hostname**
4. Fill in:
   - **Subdomain**: leave empty
   - **Domain**: `asdweq123.org`
   - **Type**: HTTP
   - **URL**: `localhost:8000`
5. Click **Save**

Now you should have **TWO** public hostnames:
- `asdweq123.org` → `http://localhost:8000`
- `api.asdweq123.org` → `http://localhost:8000`

### Step 3: Verify in Logs (No Restart Needed!)
The tunnel automatically picks up dashboard changes. Check the logs:

```bash
ssh root@runpod-server
tail -f /tmp/cloudflared.log
```

You should now see BOTH hostnames in the ingress:
```json
"ingress":[
  {"hostname":"asdweq123.org","service":"http://localhost:8000"},
  {"hostname":"api.asdweq123.org","service":"http://localhost:8000"},
  {"service":"http_status:404"}
]
```

### Step 4: Test from Mac
```bash
curl -s https://asdweq123.org/health | jq
curl -s https://api.asdweq123.org/health | jq
```

Both should return:
```json
{"status":"ok"}
```

---

## Option 2: Delete and Recreate Tunnel via CLI (If dashboard fails)

If the dashboard doesn't work or you want full control, delete the tunnel and recreate it via CLI. See `RECREATE_TUNNEL_CLI.md`.

---

## Why This Happened
- Tunnels created via **Cloudflare UI** = Dashboard-managed (ignore local config)
- Tunnels created via **CLI** = YAML-managed (use local config)

Your tunnel is dashboard-managed, so you must configure it in the dashboard, not in `~/.cloudflared/config.yml`.

---

## Next Steps
1. Add `asdweq123.org` hostname in dashboard
2. Verify both hostnames appear in logs
3. Test both endpoints from Mac
4. Update backend `.env` to use working endpoint
5. Test full chat flow from frontend
