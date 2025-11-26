# 🔧 RUNPOD PROXY FIX - Server Works, Need Access

## ✅ Current Status

**Server Status:** WORKING ✅  
**Model Loaded:** meta-llama/Meta-Llama-3.1-8B-Instruct ✅  
**Internal Test:** SUCCESS ✅

**Problem:** Can't access from Mac (proxy URL not working)

---

## 🎯 Solution: SSH Tunnel (EASIEST & MOST RELIABLE)

This creates a secure tunnel from your Mac to RunPod's port 8888.

### Step 1: Open TWO Terminals on Your Mac

**Terminal 1 - Keep SSH Connection Open:**

```bash
ssh -L 8888:localhost:8888 root@194.68.245.166 -p 22124 -i ~/.ssh/id_ed25519
```

**Keep this terminal open!** It creates the tunnel. You'll see the RunPod prompt.

---

### Step 2: Test in Terminal 2

**Open a NEW terminal on your Mac:**

```bash
# Test health
curl http://localhost:8888/health

# Should return: OK

# Test models
curl http://localhost:8888/v1/models | python3 -m json.tool

# Should show: meta-llama/Meta-Llama-3.1-8B-Instruct

# Test completion
curl http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Istanbul is famous for",
    "max_tokens": 50,
    "temperature": 0.7
  }' | python3 -m json.tool
```

**If these work, you're connected!** ✅

---

## 🌐 Solution 2: Make It Public with ngrok

If you need a public URL (for Render backend), use ngrok:

### Step 1: Install ngrok (if not installed)

```bash
brew install ngrok
```

### Step 2: Start ngrok Tunnel

**In Terminal 1 (with SSH tunnel still running):**

**In a NEW Terminal 3 on your Mac:**

```bash
ngrok http 8888
```

You'll see output like:

```
Forwarding   https://abcd-1234-5678.ngrok-free.app -> http://localhost:8888
```

**Copy this URL!** This is your public LLM endpoint.

---

### Step 3: Test ngrok URL

```bash
# Test from anywhere
curl https://abcd-1234-5678.ngrok-free.app/v1/models

# Test completion
curl https://abcd-1234-5678.ngrok-free.app/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Istanbul is famous for",
    "max_tokens": 50
  }'
```

---

## 🚀 Update Render Backend

### Option A: Using ngrok URL (if you set it up)

1. Go to https://dashboard.render.com
2. Find your backend service
3. Go to **Environment** tab
4. Update:

```
RUNPOD_LLM_ENDPOINT=https://abcd-1234-5678.ngrok-free.app
RUNPOD_LLM_MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
```

5. Click **"Manual Deploy"**

### Option B: Host backend locally (for testing)

If you want to test everything locally first:

1. Keep SSH tunnel running (Terminal 1)
2. Keep ngrok running (Terminal 3) OR just use localhost
3. Run backend locally with these env vars:

```bash
export RUNPOD_LLM_ENDPOINT=http://localhost:8888
export RUNPOD_LLM_MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
cd /Users/omer/Desktop/ai-stanbul/backend
python main_modular.py
```

---

## 📋 Summary of Terminal Setup

**Terminal 1:** SSH Tunnel (keep open)
```bash
ssh -L 8888:localhost:8888 root@194.68.245.166 -p 22124 -i ~/.ssh/id_ed25519
```

**Terminal 2:** Testing commands
```bash
curl http://localhost:8888/v1/models
```

**Terminal 3 (Optional):** ngrok for public URL
```bash
ngrok http 8888
```

**Terminal 4 (Optional):** Local backend testing
```bash
cd /Users/omer/Desktop/ai-stanbul/backend
export RUNPOD_LLM_ENDPOINT=http://localhost:8888
python main_modular.py
```

---

## ⚠️ Important Notes

1. **SSH Tunnel must stay open** - If you close Terminal 1, the connection breaks
2. **ngrok URL changes** - Each time you restart ngrok, you get a new URL (paid plan = fixed URL)
3. **RunPod proxy alternative** - If you want to use RunPod's proxy, you need to expose port 8888 in pod settings (requires pod restart)

---

## 🎯 Recommended Setup for Production

**Best approach:**

1. ✅ Use SSH tunnel for development/testing (free, reliable)
2. ✅ Use ngrok for temporary public access (free tier)
3. ✅ For permanent solution, expose port 8888 in RunPod (requires pod template edit + restart)

---

## 🔍 Alternative: Fix RunPod Proxy (Requires Restart)

If you want to use RunPod's built-in proxy:

1. Go to https://www.runpod.io/console/pods
2. Stop your current pod
3. Edit the pod template
4. Add **8888** to "Exposed HTTP Ports"
5. Start the pod again
6. Re-run all setup steps (Steps 1-8 from main guide)
7. The proxy URL will then work: `https://xxxxx-8888.proxy.runpod.net`

**⚠️ This will restart your pod and you'll lose the current running server!**

---

## ✅ Next Steps

1. **Now:** Set up SSH tunnel (Terminal 1)
2. **Test:** Run curl commands from Terminal 2
3. **Optional:** Set up ngrok for public URL (Terminal 3)
4. **Deploy:** Update Render backend with new LLM endpoint
5. **Test:** Full integration test

**Start with the SSH tunnel! It's the fastest way to get working.**
