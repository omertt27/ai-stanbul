# 🔐 Complete Llama 3.1 8B Setup with HuggingFace Token

**CRITICAL:** Llama 3.1 8B is a gated model that **requires a HuggingFace token** to download.

---

## 📋 Prerequisites Checklist

Before starting, make sure you have:

1. ✅ **HuggingFace Account** - Free at https://huggingface.co/join
2. ✅ **HuggingFace Token** - Get from https://huggingface.co/settings/tokens
3. ✅ **Accept Llama Terms** - Required at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
4. ✅ **RunPod GPU Instance** - Running with SSH access
5. ✅ **8GB+ Disk Space** - For model download

---

## 🚀 Step-by-Step Setup

### Step 1: Get Your HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name it (e.g., "runpod-llm")
4. Select **"Read"** permission
5. Click **"Generate"**
6. Copy the token (starts with `hf_...`)

### Step 2: Accept Llama 3.1 Terms

1. Go to https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. Click **"Accept"** on the gated model notice
3. Wait for approval (usually instant)

### Step 3: SSH into RunPod

```bash
ssh root@ssh.runpod.io -i ~/.ssh/id_ed25519 -p 19123
```

### Step 4: Set Your HuggingFace Token

**IMPORTANT:** Do this first in your SSH session!

```bash
export HF_TOKEN="hf_YOUR_ACTUAL_TOKEN_HERE"
```

To verify it's set:
```bash
echo $HF_TOKEN
```

Should output your token starting with `hf_`

### Step 5: Install vLLM

```bash
pip install vllm
```

### Step 6: Start the LLM Server

**Choose one method:**

#### Method A: Auto-download and start (simplest)
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 &
```

#### Method B: Pre-download then start (track progress)
```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download model (takes 5-10 minutes)
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir ./llama-3.1-8b

# Start server from local path
python -m vllm.entrypoints.openai.api_server \
  --model ./llama-3.1-8b \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 &
```

#### Method C: Persistent server (survives SSH disconnect)
```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 > /workspace/llm_server.log 2>&1 &

echo $! > /workspace/llm_server.pid
```

---

## ⏱️ Wait for Model to Load

**First time:** 5-10 minutes (model download + loading)  
**Subsequent starts:** 30-60 seconds (loading only)

### Monitor progress:

```bash
# Watch download size grow
watch -n 5 'du -sh ~/.cache/huggingface/hub'

# Check logs (if using Method C)
tail -f /workspace/llm_server.log

# Check if process is running
ps aux | grep vllm
```

### Look for this message:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8888
```

---

## ✅ Test the Server

### 1. Test locally on RunPod:

```bash
# Health check
curl http://localhost:8888/health

# Should return: "OK"

# List available models
curl http://localhost:8888/v1/models | python3 -m json.tool

# Should show: "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Test generation
curl http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Istanbul is famous for",
    "max_tokens": 50
  }'
```

### 2. Test via RunPod proxy (from your local machine):

```bash
# Test models endpoint
curl https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1/models

# Test generation
curl https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Tell me about Hagia Sophia",
    "max_tokens": 100
  }'
```

---

## 🔧 Troubleshooting

### Error: "401 Unauthorized" or "Repo is gated"

**Cause:** HuggingFace token not set or terms not accepted

**Solution:**
1. Make sure you ran: `export HF_TOKEN="hf_..."`
2. Verify token: `echo $HF_TOKEN`
3. Accept terms: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
4. Try again after 1-2 minutes

### Error: "Downloading snapshot" taking forever

**Cause:** Model is 8GB+, takes time on first download

**Solution:**
- Monitor progress: `watch -n 5 'du -sh ~/.cache/huggingface/hub'`
- Wait 5-10 minutes for first download
- Use Method B (pre-download) to track progress better

### Error: "CUDA out of memory"

**Solution:** Reduce max context length:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 2048 &  # Reduced from 4096
```

### Error: "Connection refused" on localhost:8888

**Cause:** Server not started yet or crashed

**Solution:**
1. Check if process is running: `ps aux | grep vllm`
2. Check logs: `tail -50 /workspace/llm_server.log` (if using Method C)
3. Run server in foreground to see errors:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
     --port 8888 \
     --host 0.0.0.0 \
     --trust-remote-code
   ```

### Server works locally but not via proxy

**Cause:** Proxy URL might be wrong

**Solution:**
1. Verify URL has hyphen: `ytc61lal7ag5sy-19123` (not underscore)
2. Make sure using HTTP port 8888, not HTTPS port
3. Check RunPod dashboard for correct proxy URL

---

## 🎯 After Server Works: Connect to Backend

### 1. Verify Render Environment Variable

Go to Render Dashboard → Your Backend → Environment

Make sure `LLM_API_URL` is exactly:
```
https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1
```

**Common mistakes:**
- ❌ Underscore instead of hyphen: `ytc61lal7ag5sy_19123` 
- ❌ Extra newline at end
- ❌ Missing `/v1` at end
- ❌ Wrong port or protocol

### 2. Make sure PURE_LLM_MODE is set

In Render environment variables, verify:
```
PURE_LLM_MODE=true
```

### 3. Redeploy Backend

In Render Dashboard:
1. Click **"Manual Deploy"**
2. Wait 2-3 minutes for deployment
3. Check logs for "LLM health check passed"

### 4. Test Backend Integration

```bash
# Test backend LLM health
curl https://api.aistanbul.net/api/v1/llm/health | python3 -m json.tool

# Should show: healthy: true, llm_available: true

# Test chat endpoint
curl -X POST https://api.aistanbul.net/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about the Blue Mosque in Istanbul",
    "language": "en"
  }' | python3 -m json.tool

# Should return real LLM response (not fallback "I understand...")
```

---

## 📊 System Info

**Model:** Meta-Llama-3.1-8B-Instruct  
**Size:** ~8GB download  
**VRAM:** 6-8GB required  
**RAM:** 16GB+ recommended  
**First startup:** 5-10 minutes (download + load)  
**Subsequent startups:** 30-60 seconds  
**Inference speed:** ~20-30 tokens/second

---

## 🚀 Quick Reference Commands

```bash
# Set token (do this first!)
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Start server (persistent)
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 --host 0.0.0.0 --trust-remote-code \
  --max-model-len 4096 > /workspace/llm_server.log 2>&1 &
echo $! > /workspace/llm_server.pid

# Check status
ps aux | grep vllm
tail -f /workspace/llm_server.log

# Test locally
curl http://localhost:8888/health
curl http://localhost:8888/v1/models

# Test via proxy
curl https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1/models

# Stop server
kill $(cat /workspace/llm_server.pid)
```

---

## ✅ Complete Success Checklist

- [ ] HuggingFace account created
- [ ] HuggingFace token generated
- [ ] Llama 3.1 terms accepted
- [ ] SSH into RunPod successful
- [ ] HF_TOKEN exported in SSH session
- [ ] vLLM installed
- [ ] Model download complete (8GB+)
- [ ] Server started on port 8888
- [ ] "Application startup complete" message seen
- [ ] Health check returns "OK" locally
- [ ] Models endpoint returns Llama 3.1 8B
- [ ] Proxy URL works from local machine
- [ ] Render LLM_API_URL verified (with hyphen, no newline)
- [ ] Render PURE_LLM_MODE=true set
- [ ] Backend redeployed
- [ ] Backend /api/v1/llm/health passes
- [ ] Chat endpoint returns real LLM responses

---

## 🆘 Still Having Issues?

1. **Check RunPod server logs:** `tail -100 /workspace/llm_server.log`
2. **Check Render backend logs:** Look for "LLM" or "health check"
3. **Verify environment variables:** No typos, no newlines, exact URLs
4. **Test each step:** Don't skip ahead - verify each step works before moving on
5. **Check HuggingFace status:** Token valid? Terms accepted? Model accessible?

---

**Last updated:** After diagnosing HuggingFace token requirement  
**Status:** Complete guide with all token setup steps
