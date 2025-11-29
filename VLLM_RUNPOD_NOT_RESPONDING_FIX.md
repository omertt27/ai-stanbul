# 🚨 vLLM Server Issue - RunPod Not Responding

## Problem Diagnosis

Your chat is failing because the **vLLM server on RunPod is not responding**. All endpoints return **404 Not Found**:

```bash
❌ https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/health → 404
❌ https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1/models → 404
```

## Root Cause

One of the following:
1. **RunPod pod is stopped** (most likely)
2. **vLLM server crashed or was never started**
3. **Port or endpoint configuration changed**
4. **Pod was terminated/hibernated**

## 🔧 Solution: Restart RunPod vLLM Server

### Step 1: Check RunPod Pod Status

1. Go to: https://www.runpod.io/console/pods
2. Find pod: `ytc61lal7ag5sy`
3. Check status:
   - ✅ **Running** → Continue to Step 2
   - ❌ **Stopped** → Click "Start" button, wait 1-2 minutes, then continue
   - ❌ **Terminated** → You'll need to create a new pod (see Step 5)

### Step 2: SSH into RunPod

```bash
# Using ngrok tunnel (RECOMMENDED - since you have it!)
ssh root@boarishly-umbonic-archer.ngrok-free.dev

# OR using RunPod SSH
ssh ytc61lal7ag5sy-64410fe8@ssh.runpod.io -i ~/.ssh/id_ed25519

# OR using direct TCP
ssh root@194.68.245.173 -p 22001 -i ~/.ssh/id_ed25519
```

### Step 3: Check if vLLM is Running

```bash
# Check for vLLM process
ps aux | grep vllm | grep -v grep

# Check if port 8888 is listening
lsof -i :8888

# Check recent logs
tail -50 /workspace/llm_server.log
# OR
tail -50 /root/vllm.log
```

**If vLLM is NOT running** → Continue to Step 4
**If vLLM IS running but not responding** → Skip to Step 4 (restart it)

### Step 4: Start/Restart vLLM Server

Copy and paste this complete command into RunPod terminal:

```bash
# Kill any existing vLLM processes
pkill -9 -f vllm
sleep 2

# Start vLLM server with Llama 3.1 8B
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  > /workspace/llm_server.log 2>&1 &

# Wait for server to start (takes 60-120 seconds to load model)
echo "⏳ Waiting for vLLM to start (this takes 60-120 seconds)..."
sleep 90

# Check if it's running
echo "🔍 Checking vLLM status..."
ps aux | grep vllm | grep -v grep

# Test health endpoint
echo "🏥 Testing health endpoint..."
curl http://localhost:8888/health

# Test v1 models endpoint
echo "📋 Testing models endpoint..."
curl http://localhost:8888/v1/models
```

### Step 5: Verify Server is Working

After vLLM starts, test these endpoints:

```bash
# Test health (should return healthy status)
curl http://localhost:8888/health

# Test models list
curl http://localhost:8888/v1/models

# Test actual generation (this confirms it works!)
curl http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Hello! Tell me about Istanbul in one sentence.",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Expected response** (should contain generated text about Istanbul)

### Step 6: Test from Outside (Your Computer)

Once vLLM is running inside RunPod, test the external URL:

```bash
# Test health via RunPod proxy
curl https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/health

# Should return: {"status":"healthy",...}
```

**If still 404** → The RunPod proxy URL might have changed. Check RunPod dashboard for the current URL.

### Step 7: Update Render Environment Variable (if URL changed)

If you got a new RunPod proxy URL:

1. Go to: https://dashboard.render.com
2. Select your backend service
3. Go to: **Environment** tab
4. Update `LLM_API_URL` with new URL (must end with `/v1`)
5. **Save Changes** (will trigger redeploy)

## 🎯 Quick Diagnosis Commands

Run these on RunPod to diagnose issues:

```bash
# Check if vLLM is running
ps aux | grep vllm | grep -v grep

# Check port
lsof -i :8888

# Check logs (last 50 lines)
tail -50 /workspace/llm_server.log

# Check GPU availability
nvidia-smi

# Check disk space
df -h

# Check if model is downloaded
ls -lh /root/.cache/huggingface/hub/
```

## 🔄 Alternative: Use ngrok Tunnel for vLLM

Since you have ngrok set up, you can expose vLLM through it:

```bash
# On RunPod (in a separate terminal or background)
ngrok http 8888

# This will give you a URL like:
# https://xyz-abc.ngrok-free.app

# Then update Render LLM_API_URL to:
# https://xyz-abc.ngrok-free.app/v1
```

## 📋 Checklist

- [ ] RunPod pod is **Running** (not stopped)
- [ ] vLLM process is active (`ps aux | grep vllm`)
- [ ] Port 8888 is listening (`lsof -i :8888`)
- [ ] Health endpoint works locally (`curl localhost:8888/health`)
- [ ] External URL works (via RunPod proxy or ngrok)
- [ ] Render `LLM_API_URL` is correct and ends with `/v1`
- [ ] Render backend redeployed after env variable change

## 🚨 If Nothing Works

### Option A: Use OpenAI as Fallback

Temporary fix while debugging RunPod:

1. Go to Render dashboard → Environment
2. Change `LLM_PROVIDER` to `openai`
3. Add `OPENAI_API_KEY` with your OpenAI key
4. Save (triggers redeploy)

### Option B: Use Local LLM

If you have a powerful Mac:

```bash
# Start Ollama locally
ollama serve

# Pull Llama 3.1
ollama pull llama3.1:8b

# Update Render LLM_API_URL to your ngrok tunnel:
# LLM_API_URL=https://your-ngrok-url.ngrok-free.app/v1
```

## 💡 Why This Happens

1. **RunPod hibernates pods** after inactivity (to save costs)
2. **vLLM crashes** if it runs out of memory
3. **Model loading fails** if disk is full
4. **RunPod restarts pod** which kills the vLLM process

## 🎯 Permanent Solution

Add a startup script on RunPod to auto-start vLLM:

```bash
# Create startup script
cat > /workspace/start_vllm.sh << 'EOF'
#!/bin/bash
pkill -9 -f vllm
sleep 2
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  > /workspace/llm_server.log 2>&1 &
EOF

chmod +x /workspace/start_vllm.sh

# Add to RunPod custom start command (in pod settings):
/workspace/start_vllm.sh
```

## 📊 Current Status

- ✅ Frontend: Working perfectly
- ✅ Backend API: Healthy
- ✅ Database: Connected
- ❌ **vLLM Server: NOT RESPONDING** ← This is the issue!

## Next Steps

1. **SSH into RunPod NOW**
2. **Run the vLLM start command** (Step 4)
3. **Wait 2 minutes for model to load**
4. **Test the health endpoint**
5. **Try your chat again**

The chat should work immediately once vLLM is running!
