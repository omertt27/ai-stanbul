# ✅ Verify Service Configuration in Dashboard

## Good News!
You have both hostnames in the dashboard:
- ✅ `api.asdweq123.org`
- ✅ `asdweq123.org`

## Now Check the Service Configuration

### Step 1: Click Edit on `api.asdweq123.org`

1. In the **Public Hostname** section, find the row for `api.asdweq123.org`
2. Click the **Edit** button (pencil icon) or the **3 dots menu** → **Edit**

### Step 2: Verify Service Settings

You should see a form with these fields:

| Field | Current Value? | Expected Value |
|-------|---------------|----------------|
| **Subdomain** | `api` | `api` ✅ |
| **Domain** | `asdweq123.org` | `asdweq123.org` ✅ |
| **Path** | (empty) | (empty) ✅ |
| **Type** | ??? | `HTTP` ← Check this! |
| **URL** | ??? | `localhost:8000` ← Check this! |

### Step 3: Update if Needed

If **Type** is not `HTTP` or **URL** is not `localhost:8000`:

1. Change **Type** to: `HTTP`
2. Change **URL** to: `localhost:8000`
3. Click **Save hostname**

### Step 4: Repeat for `asdweq123.org`

1. Click **Edit** on `asdweq123.org`
2. Verify:
   - **Type**: `HTTP`
   - **URL**: `localhost:8000`
3. Save if needed

---

## Test After Saving

Once both hostnames have the correct service configuration, test from your Mac:

```bash
# Test api subdomain
curl -s https://api.asdweq123.org/health | jq

# Test root domain
curl -s https://asdweq123.org/health | jq
```

Both should return:
```json
{"status":"ok"}
```

---

## Expected Log Output on RunPod

After saving in the dashboard, check the tunnel logs on RunPod:

```bash
ssh root@runpod-server
tail -30 /workspace/cloudflared.log | grep -i ingress
```

You should now see **BOTH** hostnames:
```json
"ingress":[
  {"hostname":"api.asdweq123.org","service":"http://localhost:8000"},
  {"hostname":"asdweq123.org","service":"http://localhost:8000"},
  {"service":"http_status:404"}
]
```

---

## What to Look For

When you click **Edit** on each hostname, you're looking for:

```
┌─────────────────────────────────────────┐
│ Edit Public Hostname                     │
├─────────────────────────────────────────┤
│ Subdomain: [api          ]               │
│ Domain:    [asdweq123.org]               │
│ Path:      [           ]                 │
├─────────────────────────────────────────┤
│ Service                                  │
│ Type: [HTTP ▼]  ← Should be HTTP         │
│ URL:  [localhost:8000]  ← Should be this │
├─────────────────────────────────────────┤
│ [Cancel]  [Save hostname]                │
└─────────────────────────────────────────┘
```

**Important:** Do NOT use `https://`, just `localhost:8000`

---

## Next Steps

1. ✅ Click Edit on `api.asdweq123.org` - verify service is `HTTP` → `localhost:8000`
2. ✅ Click Edit on `asdweq123.org` - verify service is `HTTP` → `localhost:8000`
3. ✅ Save any changes
4. ✅ Test both endpoints from Mac
5. ✅ Check RunPod logs to confirm both ingress rules are active

Let me know what you see when you click **Edit** on each hostname! 🚀
