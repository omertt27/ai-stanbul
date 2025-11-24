# 🚀 RunPod Fresh Start - Complete Setup Instructions

**Your RunPod restarted, here's everything you need to do from scratch.**

---

## 🔍 What Happened

Your previous attempt failed with: `OSError: [Errno 122] Disk quota exceeded`

**The Llama 3.1 8B model needs ~16GB download + cache space = 25GB+ total**

---

## ✅ Step-by-Step Instructions

### Step 1: SSH into RunPod

**⚠️ IMPORTANT: Run on YOUR LOCAL TERMINAL (Mac), not inside RunPod!**

**Option A: SSH via RunPod Proxy (Recommended)**
```bash
ssh vn290bqt32835t-64410fd1@ssh.runpod.io -i ~/.ssh/id_ed25519
```

**Option B: Direct TCP Connection (Supports SCP & SFTP)**
```bash
ssh root@194.68.245.166 -p 22124 -i ~/.ssh/id_ed25519
```

**Option C: If SSH key not found, try without `-i` flag:**
```bash
ssh vn290bqt32835t-64410fd1@ssh.runpod.io
```

Use Option A for general terminal access. Use Option B if you need to transfer files.

---

### Step 2: Check Available Disk Space

**CRITICAL: Check this FIRST before downloading anything!**

```bash
df -h
```

Look for available space:
- `/workspace` - Pod storage (persistent, limited)
- `/root` - Container storage (temporary, usually larger)

**You need at least 25GB free in ONE location.**

---

### Step 3A: If You Have 25GB+ Free

Great! Proceed with Llama 3.1 8B.

#### Set Environment Variables

```bash
# Set HuggingFace token (REQUIRED)
export HF_TOKEN="hf_YOUR_ACTUAL_TOKEN_HERE"

# Use the disk with most space (check df -h output)
# Option 1: Use /root if it has more space
export HF_HOME=/root/.cache/huggingface

# Option 2: Use /workspace if it has more space
# export HF_HOME=/workspace/.cache/huggingface
```

#### Verify Token is Set

```bash
echo $HF_TOKEN
# Should output: hf_...
```

#### Install Required Packages

```bash
pip install vllm huggingface_hub hf-transfer
```

#### Login to HuggingFace (REQUIRED)

```bash
huggingface-cli login --token $HF_TOKEN
```

You should see:
```
Token is valid (permission: fineGrained or read).
Your token has been saved to /workspace/.cache/huggingface/token
Login successful
```

#### Download Model from HuggingFace

```bash
# Disable fast transfer to avoid issues
unset HF_HUB_ENABLE_HF_TRANSFER

# Download model first (track progress) - takes 5-10 minutes
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir /workspace/llama-3.1-8b

# You'll see progress for each file being downloaded
```

#### Start Server from Downloaded Model

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/llama-3.1-8b \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 &
```

**Benefits of downloading first:**
- ✅ See download progress clearly
- ✅ Can verify download completed before starting server
- ✅ Easier to troubleshoot disk space issues
- ✅ Can resume if interrupted
  --trust-remote-code \
  --max-model-len 4096 &
```

#### Monitor Server Startup (30-60 seconds after download)

The model is already downloaded, so server should start quickly!

```bash
# Check process is running
ps aux | grep vllm

# Check disk usage
df -h
```

#### Wait for "Application startup complete"

Look for this in output:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8888
```

#### Test Locally

```bash
curl http://localhost:8888/health
# Should return: "OK"

curl http://localhost:8888/v1/models
# Should show: meta-llama/Meta-Llama-3.1-8B-Instruct
```

---

### Step 3B: If You Have LESS Than 25GB Free

**Use Qwen 2.5 7B instead - only needs 15GB total, equally good!**

#### Set Environment Variables

```bash
# Set HuggingFace token (REQUIRED)
export HF_TOKEN="hf_YOUR_ACTUAL_TOKEN_HERE"

# Use the disk with most space
export HF_HOME=/root/.cache/huggingface
```

#### Install Required Packages

```bash
pip install vllm huggingface_hub
```

#### Login to HuggingFace

```bash
huggingface-cli login --token $HF_TOKEN
```

#### Download Qwen Model from HuggingFace

```bash
# Download model first (track progress)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /workspace/qwen-2.5-7b

# This will take 3-5 minutes (smaller than Llama)
```

#### Start Server with Qwen

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 &
```

**Qwen 2.5 7B advantages:**
- ✅ Smaller download (8GB vs 16GB)
- ✅ Less disk space (15GB total vs 25GB)
- ✅ Faster download time
- ✅ Equal or better performance for chat
- ✅ Better multilingual support

#### Monitor and Test

Same as Step 3A above.

---

### Step 4: Make Server Persistent (Survives SSH Disconnect)

Once server is running and tested, make it persistent:

```bash
# Kill the background process first
pkill -f vllm

# Restart with nohup
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export HF_HOME=/root/.cache/huggingface  # Or wherever you chose

nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 > /workspace/llm_server.log 2>&1 &

echo $! > /workspace/llm_server.pid
```

Or for Qwen:
```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 > /workspace/llm_server.log 2>&1 &

echo $! > /workspace/llm_server.pid
```

---

### Step 5: Test from Your Local Machine

```bash
# Test via RunPod proxy
curl https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1/models

# Test generation
curl https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Istanbul is famous for",
    "max_tokens": 50
  }'
```

---

### Step 6: Update Render Backend

#### 6.1 Check Environment Variables

Go to Render Dashboard → Your Backend → Environment

Make sure these are set:

```
LLM_API_URL=https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1
PURE_LLM_MODE=true
```

**Common mistakes to avoid:**
- ❌ Underscore instead of hyphen in URL
- ❌ Extra newline at end of URL
- ❌ Missing `/v1` at end
- ❌ PURE_LLM_MODE not set to true

#### 6.2 Redeploy Backend

In Render Dashboard:
1. Click **"Manual Deploy"**
2. Wait 2-3 minutes
3. Check logs for "LLM health check passed"

#### 6.3 Test Backend Integration

```bash
# Test backend LLM health
curl https://api.aistanbul.net/api/v1/llm/health | python3 -m json.tool

# Should show:
# {
#   "healthy": true,
#   "llm_available": true,
#   ...
# }

# Test chat endpoint
curl -X POST https://api.aistanbul.net/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about the Blue Mosque in Istanbul",
    "language": "en"
  }' | python3 -m json.tool

# Should return real LLM response (NOT "I understand...")
```

---

## 🔧 Troubleshooting After Restart

### Issue: "Disk quota exceeded" again

**Solution:** You need more space!

```bash
# Check what's using space
du -sh /workspace/* | sort -h
du -sh /root/.cache/* | sort -h

# Clear old HuggingFace cache
rm -rf /workspace/.cache/huggingface/hub
rm -rf /root/.cache/huggingface/hub

# Or switch to smaller model (Qwen 2.5 7B)
```

### Issue: "401 Unauthorized"

**Solution:** HF token not set or terms not accepted

```bash
# Set token again (environment is lost after restart)
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Verify
echo $HF_TOKEN

# Make sure you accepted terms at:
# https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
```

### Issue: "No module named 'vllm'"

**Solution:** Reinstall (lost after restart if not persistent)

```bash
pip install vllm
```

### Issue: Server not responding on localhost:8888

**Solution:** Check if process is running

```bash
# Check process
ps aux | grep vllm

# If not running, check logs
cat /workspace/llm_server.log

# If running, test with curl
curl http://localhost:8888/health
```

### Issue: Model download very slow

**Solution:** Normal for first time, monitor progress

```bash
# Check download progress
watch -n 5 'du -sh $HF_HOME/hub'

# Or switch to faster mirror (if available)
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 📊 Disk Space Requirements

| Model | Download Size | Total Space Needed | Performance |
|-------|--------------|-------------------|-------------|
| Llama 3.1 8B | ~16GB | ~25GB | Excellent |
| Qwen 2.5 7B | ~8GB | ~15GB | Excellent (better multilingual) |
| Qwen 2.5 3B | ~4GB | ~8GB | Good (fastest) |

---

## ✅ Complete Checklist

### Pre-flight
- [ ] SSH into RunPod successful
- [ ] Checked disk space with `df -h`
- [ ] Have 25GB+ free (or using Qwen with 15GB+)

### Setup
- [ ] `HF_TOKEN` exported
- [ ] `HF_HOME` set to disk with most space
- [ ] Token verified with `echo $HF_TOKEN`
- [ ] vLLM installed with `pip install vllm`

### Server Start
- [ ] Server command executed
- [ ] Model downloading (monitor with `df -h` or `du -sh`)
- [ ] "Application startup complete" message seen
- [ ] Local health check passes: `curl http://localhost:8888/health`
- [ ] Local models endpoint works

### Persistent Setup
- [ ] Server restarted with `nohup`
- [ ] PID saved to `/workspace/llm_server.pid`
- [ ] Logs writing to `/workspace/llm_server.log`

### External Testing
- [ ] Proxy URL works from local machine
- [ ] Generation test successful

### Backend Integration
- [ ] Render `LLM_API_URL` verified (with hyphen, no newline)
- [ ] Render `PURE_LLM_MODE=true` set
- [ ] Backend redeployed
- [ ] Backend `/api/v1/llm/health` passes
- [ ] Chat endpoint returns real LLM responses

---

## 🚀 Quick Copy-Paste Commands

### For 25GB+ Space (Llama 3.1 8B)

```bash
# SSH into RunPod
ssh root@ssh.runpod.io -i ~/.ssh/id_ed25519 -p 19123

# Check space
df -h

# Set environment
export HF_TOKEN="hf_YOUR_ACTUAL_TOKEN_HERE"
export HF_HOME=/workspace/.cache/huggingface

# Install packages
pip install vllm huggingface_hub hf-transfer

# Login to HuggingFace
huggingface-cli login --token $HF_TOKEN

# Disable fast transfer (can cause issues)
unset HF_HUB_ENABLE_HF_TRANSFER

# Download model (5-10 minutes)
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir /workspace/llama-3.1-8b

# Start server
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 --host 0.0.0.0 --trust-remote-code \
  --max-model-len 4096 > /workspace/llm_server.log 2>&1 &

echo $! > /workspace/llm_server.pid

# Monitor (wait for "Application startup complete")
tail -f /workspace/llm_server.log

# Test (after startup completes)
curl http://localhost:8888/health
curl http://localhost:8888/v1/models
```

### For 15GB+ Space (Qwen 2.5 7B) - RECOMMENDED

```bash
# SSH into RunPod
ssh root@ssh.runpod.io -i ~/.ssh/id_ed25519 -p 19123

# Check space
df -h

# Set environment
export HF_TOKEN="hf_YOUR_ACTUAL_TOKEN_HERE"
export HF_HOME=/root/.cache/huggingface

# Install and start
pip install vllm

nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8888 --host 0.0.0.0 --trust-remote-code \
  --max-model-len 4096 > /workspace/llm_server.log 2>&1 &

echo $! > /workspace/llm_server.pid

# Monitor
tail -f /workspace/llm_server.log

# Test (after "Application startup complete")
curl http://localhost:8888/health
```

---

## 💡 Pro Tips

1. **Use Qwen if disk space is tight** - it's actually better for multilingual chat!
2. **Monitor disk space during download** - if it fills up, kill process immediately
3. **Set `HF_HOME` to the disk with most space** - check `df -h` first
4. **Save your HF_TOKEN somewhere** - you'll need to set it again after each restart
5. **Use `nohup` and save PID** - server will survive SSH disconnect
6. **Check logs regularly** - `tail -f /workspace/llm_server.log`

---

**Ready? Start with Step 1 and work through each step carefully!** 🚀

**Recommendation: Use Qwen 2.5 7B - it needs less space and is actually better for your use case!**
