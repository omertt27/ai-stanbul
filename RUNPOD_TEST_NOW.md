# 🧪 RUNPOD TESTING - DO THIS NOW

## Step 1: Test Inside RunPod (in your SSH terminal)

```bash
# Test health endpoint
curl http://localhost:8888/health

# Test models endpoint
curl http://localhost:8888/v1/models

# Test a simple completion
curl http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Hello",
    "max_tokens": 10
  }'
```

**If these work, the server is fine. The issue is the proxy URL.**

---

## Step 2: Get the Correct Proxy URL

### Option A: Check RunPod Dashboard

1. Go to https://www.runpod.io/console/pods
2. Find your running pod
3. Look for **"Connect"** button
4. Find the **HTTP Service** section
5. Look for port **8888**
6. Copy the full URL (should look like: `https://xxxxx-8888.proxy.runpod.net`)

### Option B: Check in RunPod Terminal

```bash
# Get your pod ID
curl -s http://169.254.169.254/latest/meta-data/public-ipv4

# Or check environment
env | grep RUNPOD
```

---

## Step 3: Verify Port is Exposed

In RunPod, you need to expose port 8888. Check if it's in your pod's exposed ports:

1. Go to RunPod dashboard
2. Click on your pod
3. Look for **"Exposed Ports"** or **"HTTP Services"**
4. Make sure **8888** is listed

**If 8888 is NOT exposed, you need to:**

1. Stop the pod
2. Edit template
3. Add port 8888 to exposed ports
4. Restart pod
5. Re-run all setup steps

---

## Step 4: Alternative - Use Direct TCP Connection

If HTTP proxy doesn't work, use direct TCP:

### On Your Mac:

```bash
# Create SSH tunnel for port 8888
ssh -L 8888:localhost:8888 root@194.68.245.166 -p 22124 -i ~/.ssh/id_ed25519
```

**Keep this terminal open!**

### In a NEW Mac terminal:

```bash
# Now test via localhost
curl http://localhost:8888/health

curl http://localhost:8888/v1/models

curl http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Tell me about Istanbul in one sentence:",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

---

## Step 5: Update Backend to Use Tunnel

If the SSH tunnel works, you can:

1. Keep the tunnel running on your Mac
2. Use `ngrok` or similar to expose it publicly
3. Or set up a permanent tunnel solution

### Quick ngrok setup:

```bash
# Install ngrok (if not installed)
brew install ngrok

# Start tunnel
ngrok http 8888
```

This will give you a public URL like: `https://xxxx-xxxx.ngrok-free.app`

Then update Render backend:
```
RUNPOD_LLM_ENDPOINT=https://xxxx-xxxx.ngrok-free.app
```

---

## 🎯 What to Do Now:

1. **First**: Test inside RunPod (Step 1) to confirm server works
2. **Second**: Get correct proxy URL from RunPod dashboard (Step 2)
3. **Third**: If proxy doesn't work, use SSH tunnel (Step 4)

**Copy the output of Step 1 tests and paste them here so I can help debug!**
