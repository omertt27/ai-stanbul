# 🔧 Fix LLM Endpoint - Final Step

## ✅ Database Migration: COMPLETE!

The `photo_url` and `photo_reference` columns have been successfully added to the production database.

---

## ⚠️ Current Issue: Wrong LLM Endpoint

Your Render backend is pointing to the **wrong RunPod endpoint**:

**Current (WRONG):**
```
LLM_API_URL=https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1
```

**Should be (CORRECT - Port 8000):**
```
LLM_API_URL=https://nnqisfv2zk46t2-8000.proxy.runpod.net/v1
```

---

## 🚀 Steps to Fix

### **Step 1: Get Your Correct RunPod Endpoint**

1. Go to: https://www.runpod.io/console/pods
2. Click your pod: **nnqisfv2zk46t2**
3. Go to **"Connect"** tab
4. Look for **"HTTP Ports [8000]"** section
5. Copy the URL (should look like: `https://nnqisfv2zk46t2-8000.proxy.runpod.net`)

---

### **Step 2: Update Render Environment Variable**

1. Go to: https://dashboard.render.com
2. Select your backend service: **ai-stanbul**
3. Go to **"Environment"** tab
4. Find `LLM_API_URL` variable
5. Click **"Edit"**
6. Update the value to:
   ```
   https://nnqisfv2zk46t2-8000.proxy.runpod.net/v1
   ```
   (Replace with your actual endpoint from Step 1, and add `/v1` at the end)
7. Click **"Save Changes"**

---

### **Step 3: Redeploy Backend**

After updating the environment variable:

1. Render will automatically trigger a redeploy
2. Wait for deployment to complete (~2-3 minutes)
3. Watch the logs for any errors

---

### **Step 4: Verify vLLM is Running on RunPod**

SSH into your RunPod pod and check:

```bash
# Check if vLLM is running on port 8000
lsof -i :8000

# Or check all Python processes
ps aux | grep vllm

# Test locally
curl http://localhost:8000/health
```

**If vLLM is NOT running**, start it:

```bash
# Start vLLM in the background with screen
screen -S vllm-server

# Inside screen, start vLLM:
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8000 \
  --host 0.0.0.0

# Wait for "Application startup complete"
# Press: Ctrl+A, then D (to detach)
```

---

### **Step 5: Test Public vLLM Endpoint**

From your Mac terminal:

```bash
# Test health
curl https://nnqisfv2zk46t2-8000.proxy.runpod.net/health

# Expected: {"status": "ok"} or similar

# Test chat completion
curl -X POST https://nnqisfv2zk46t2-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "messages": [
      {"role": "user", "content": "Say hello in Turkish"}
    ],
    "max_tokens": 50
  }'

# Should return JSON with AI-generated text
```

---

### **Step 6: Test Backend Integration**

Once Render redeploys with the new endpoint:

```bash
# Test the chat API
curl -X POST https://ai-stanbul.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the best places to visit in Istanbul?",
    "context": {}
  }' | python3 -m json.tool
```

**Expected:** Real AI-generated response (not fallback text)

---

### **Step 7: Test Frontend**

1. Go to: https://aistanbul.net
2. Open the chat
3. Ask: "What are the best restaurants in Sultanahmet?"
4. **You should now get real AI responses!** 🎉

---

## 🔍 Troubleshooting

### Issue: "Connection timeout" or "Unable to connect to LLM"

**Check:**
1. Is vLLM running on RunPod? (`ps aux | grep vllm`)
2. Is it listening on port 8000? (`lsof -i :8000`)
3. Can you access it locally? (`curl http://localhost:8000/health`)
4. Is the public endpoint correct? (Check RunPod console)

### Issue: Still getting fallback responses

**Check:**
1. Did Render finish redeploying? (Check deployment logs)
2. Is `LLM_API_URL` updated? (Check Environment tab)
3. Are there errors in Render logs? (Check Logs tab)

### Issue: vLLM not responding

```bash
# Check GPU memory
nvidia-smi

# Restart vLLM if needed
pkill -f vllm
screen -S vllm-server
# ... start vLLM again
```

---

## ✅ Success Checklist

- [ ] Database migration complete (`photo_url` and `photo_reference` added)
- [ ] vLLM running on RunPod port 8000
- [ ] Public vLLM endpoint accessible
- [ ] Render `LLM_API_URL` updated to correct endpoint
- [ ] Render backend redeployed successfully
- [ ] Backend can call LLM API (no connection errors in logs)
- [ ] Frontend chat returns real AI responses

---

## 📊 Current Status

**✅ FIXED:**
- Database schema error (missing columns)
- Direct database connection established
- Migration script works

**⚠️ NEEDS FIX:**
- Update Render `LLM_API_URL` environment variable
- Verify vLLM is running on port 8000
- Test end-to-end chat functionality

---

## 🎯 What Port Does Your vLLM Use?

**Important:** Tell me which port your vLLM is actually running on:

- Port **8000** (recommended, Jupyter on 8888)
- Port **8888** (if Jupyter is disabled)
- Other port?

Then I'll give you the exact endpoint URL to use!

---

**Need help?** Check [RUNPOD_VLLM_SETUP.md](./RUNPOD_VLLM_SETUP.md) for vLLM setup.
