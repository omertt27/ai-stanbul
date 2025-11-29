# 🎯 URGENT: Start vLLM on Your New RunPod

## Current Situation

- ✅ RunPod pod is **running**: `oln8fcw6x2t614`
- ❌ vLLM is **NOT running** - Port 8888 shows Jupyter Lab instead
- ❌ Chat is failing because no LLM server is available

## 🚀 IMMEDIATE ACTION REQUIRED

### Step 1: SSH into RunPod

Choose ONE of these methods:

```bash
# Option A: Standard SSH (recommended)
ssh oln8fcw6x2t614-64410d62@ssh.runpod.io -i ~/.ssh/id_ed25519

# Option B: Direct TCP SSH
ssh root@194.68.245.153 -p 22048 -i ~/.ssh/id_ed25519

# Option C: Use RunPod Web Terminal
# Go to: https://www.runpod.io/console/pods
# Click "Connect" → "Open Web Terminal"
```

### Step 2: Start vLLM Server

**Copy and paste this ENTIRE command** into RunPod terminal:

```bash
# Kill any existing vLLM processes
pkill -9 -f vllm
sleep 2

# Start vLLM server on port 8000 (different from Jupyter's 8888)
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  > /workspace/llm_server.log 2>&1 &

echo "✅ vLLM started! Waiting 90 seconds for model to load..."
sleep 90

# Check if it's running
echo "🔍 Checking vLLM status..."
ps aux | grep vllm | grep -v grep

# Test health endpoint
echo "🏥 Testing health endpoint..."
curl http://localhost:8000/health

# Test models endpoint
echo "📋 Testing models endpoint..."
curl http://localhost:8000/v1/models
```

**Wait for the commands to complete!** Model loading takes 60-120 seconds.

### Step 3: Verify vLLM is Working

Test inside RunPod:

```bash
# Should return: OK or similar
curl http://localhost:8000/health

# Should return model list
curl http://localhost:8000/v1/models

# Test actual generation
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Hello! Tell me about Istanbul.",
    "max_tokens": 50
  }'
```

If you see generated text about Istanbul → ✅ **vLLM is working!**

### Step 4: Update Render Environment Variable

Now update your Render backend with the CORRECT URL:

1. Go to: https://dashboard.render.com
2. Select your **backend** service
3. Click **Environment** tab
4. Find or add: `LLM_API_URL`
5. Set value to:

```
https://oln8fcw6x2t614-8000.proxy.runpod.net/v1
```

**IMPORTANT:** Note the port is **8000** (not 8888, not 19123)

6. Click **Save Changes**
7. Wait for Render to redeploy (2-3 minutes)

### Step 5: Test Your Chat

Once Render finishes redeploying:

1. Go to: https://aistanbul.net
2. Open chat
3. Type: "hi" or "tell me about Istanbul"
4. Should get a proper AI response! 🎉

## 📋 Quick Reference

### Your RunPod Configuration

- **Pod ID:** `oln8fcw6x2t614`
- **vLLM Port:** 8000
- **Jupyter Port:** 8888 (ignore this)
- **SSH:** `ssh oln8fcw6x2t614-64410d62@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **Direct IP:** `194.68.245.153:22048`

### Correct URLs

✅ **LLM API URL (for Render):**
```
https://oln8fcw6x2t614-8000.proxy.runpod.net/v1
```

✅ **Health Check:**
```
https://oln8fcw6x2t614-8000.proxy.runpod.net/health
```

❌ **Wrong URLs (don't use these):**
```
https://oln8fcw6x2t614-19123.proxy.runpod.net/... (web terminal)
https://oln8fcw6x2t614-8888.proxy.runpod.net/...  (jupyter lab)
```

## 🔍 Troubleshooting

### If vLLM doesn't start:

```bash
# Check logs
tail -100 /workspace/llm_server.log

# Check GPU
nvidia-smi

# Check if model is downloaded
ls -lh /root/.cache/huggingface/hub/

# Check disk space
df -h
```

### If "model not found" error:

The model might not be downloaded. Download it first:

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Meta-Llama-3.1-8B-Instruct')"
```

### If "out of memory" error:

Reduce memory usage:

```bash
pkill -9 -f vllm
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  > /workspace/llm_server.log 2>&1 &
```

## 🎯 Testing from Your Computer

After vLLM is running and Render is updated:

```bash
# Test health (should return OK or healthy)
curl https://oln8fcw6x2t614-8000.proxy.runpod.net/health

# Test models list
curl https://oln8fcw6x2t614-8000.proxy.runpod.net/v1/models

# Test generation
curl https://oln8fcw6x2t614-8000.proxy.runpod.net/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "What are the best places to visit in Istanbul?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## 💡 Why Port 8000 Instead of 8888?

- **Port 8888:** Used by Jupyter Lab (comes with RunPod template)
- **Port 8000:** Standard vLLM port, won't conflict with Jupyter

## 🎯 One-Line Command (Copy This!)

If you want the absolute fastest way, paste this single line into RunPod:

```bash
pkill -9 -f vllm && sleep 2 && nohup python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000 --host 0.0.0.0 --max-model-len 8192 --gpu-memory-utilization 0.9 > /workspace/llm_server.log 2>&1 & sleep 90 && curl http://localhost:8000/health
```

## 📊 Expected Timeline

1. **SSH into RunPod:** 10 seconds
2. **Start vLLM command:** 5 seconds
3. **Model loading:** 90-120 seconds ⏳
4. **Test working:** 5 seconds
5. **Update Render:** 30 seconds
6. **Render redeploy:** 2-3 minutes ⏳
7. **Test chat:** 5 seconds

**Total:** ~5-7 minutes to fully working chat! 🎉

## ✅ Success Criteria

You'll know it's working when:

1. ✅ `curl localhost:8000/health` returns OK
2. ✅ `curl localhost:8000/v1/models` returns model list
3. ✅ External URL works: `curl https://oln8fcw6x2t614-8000.proxy.runpod.net/health`
4. ✅ Render shows `LLM_API_URL=https://oln8fcw6x2t614-8000.proxy.runpod.net/v1`
5. ✅ Chat on aistanbul.net responds with AI-generated content

## 🚨 If Still Not Working

Share the output of:

```bash
# On RunPod
tail -100 /workspace/llm_server.log
ps aux | grep vllm
curl http://localhost:8000/health
```

And I can help debug further!

---

**TL;DR:**
1. SSH into RunPod
2. Run the vLLM start command (Step 2)
3. Update Render with: `https://oln8fcw6x2t614-8000.proxy.runpod.net/v1`
4. Wait for redeploy
5. Test chat ✅
