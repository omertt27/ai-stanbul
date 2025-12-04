# 🔍 Check if vLLM is Running on RunPod

## Your Current Situation

**Your RunPod URL**: `https://4r1su4zfuok0s7-19123.proxy.runpod.net/i1e3hjxfw3uwhxwp4nng62k7gmzjpjfo/`

**Problem**: This URL returns a web terminal (HTML), not vLLM API responses
**Diagnosis**: ❌ **vLLM is NOT running on your pod**

## Step 1: Access RunPod Console

1. Open: https://www.runpod.io/console/pods
2. Log in to your RunPod account
3. You should see your pod: `4r1su4zfuok0s7`
4. Check status:
   - 🟢 **Running** = Pod is on
   - 🔴 **Stopped** = Need to start it first
   - 💤 **Sleeping** = Need to wake it up

## Step 2: Get SSH Connection Details

### Option A: Web Terminal (Easiest)
1. Click on your pod `4r1su4zfuok0s7`
2. Click **"Connect"** button
3. Select **"Start Web Terminal"** or **"HTTP Terminal"**
4. A terminal will open in your browser
5. Skip to Step 3

### Option B: SSH from Your Mac (Better)
1. Click on your pod `4r1su4zfuok0s7`
2. Click **"Connect"** button
3. Look for **"SSH over exposed TCP"**
4. You'll see something like:
   ```
   ssh root@4r1su4zfuok0s7-ssh.proxy.runpod.net -p 12345
   ```
5. Copy this command
6. Open Terminal on your Mac
7. Paste and run the SSH command
8. If asked "Are you sure?", type: `yes`

## Step 3: Check if vLLM is Running

Once you're in the RunPod terminal (web or SSH), run these commands:

```bash
# Check if vLLM process is running
ps aux | grep vllm
```

### Possible Results:

#### ✅ vLLM IS Running
You'll see something like:
```
root  1234  python -m vllm.entrypoints.openai.api_server --model /workspace/...
```
**Action**: Skip to Step 5 (verify port)

#### ❌ vLLM is NOT Running
You'll only see:
```
root  5678  grep vllm
```
**Action**: Continue to Step 4 (start vLLM)

## Step 4: Start vLLM Server

If vLLM is not running, start it with these commands:

```bash
# Navigate to workspace
cd /workspace

# Check what models are available
ls -lh

# Start vLLM server (replace MODEL_NAME with your actual model)
# Common model names:
# - Meta-Llama-3.1-8B-Instruct-AWQ-INT4
# - Meta-Llama-3-8B-Instruct
# - llama-3.1-8b-instruct

nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --port 8888 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &

# Note: If the model name is different, adjust the --model path
```

### Watch vLLM Startup Logs

```bash
# Watch the logs in real-time
tail -f /workspace/vllm.log

# You should see:
# - Loading model...
# - Loading weights...
# - Application startup complete
# 
# This takes 2-3 minutes
# Press Ctrl+C when you see "Application startup complete"
```

### If Model Not Found

If you get an error like "Model not found", check available models:

```bash
# List all files in workspace
ls -la /workspace/

# Look for model directories
find /workspace -name "*llama*" -o -name "*Llama*" 2>/dev/null

# Common locations:
# /workspace/models/
# /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/
# /workspace/llama-3.1-8b/
```

Then use the correct path in the `--model` argument.

## Step 5: Verify vLLM is Running

### Check the Process
```bash
# Verify vLLM is running
ps aux | grep vllm

# Should show a python process
```

### Check the Port
```bash
# Verify port 8888 is listening
netstat -tuln | grep 8888

# Should show:
# tcp   0   0.0.0.0:8888   0.0.0.0:*   LISTEN
```

### Test Locally on Pod
```bash
# Test health endpoint from inside the pod
curl http://localhost:8888/health

# Expected response:
# {"status":"healthy","model":"..."}
```

### Test vLLM API
```bash
# Test the models endpoint
curl http://localhost:8888/v1/models

# Test a simple completion
curl -X POST http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "prompt": "Hello!",
    "max_tokens": 20
  }'
```

## Step 6: Find Your Public vLLM Endpoint

Once vLLM is running on port 8888, you need to find the public URL:

### Option A: Check RunPod Console
1. In RunPod console, click your pod
2. Look for **"HTTP Ports"** or **"Exposed Ports"**
3. Find port **8888**
4. The public URL will be something like:
   ```
   https://4r1su4zfuok0s7-8888.proxy.runpod.net
   ```

### Option B: Construct the URL
Your pod ID is: `4r1su4zfuok0s7`
Port is: `8888`
Format: `https://POD-ID-PORT.proxy.runpod.net`

**Your vLLM endpoint should be**:
```
https://4r1su4zfuok0s7-8888.proxy.runpod.net
```

## Step 7: Test Public Endpoint

From your Mac terminal (not RunPod SSH):

```bash
# Test health
curl https://4r1su4zfuok0s7-8888.proxy.runpod.net/health

# Test models
curl https://4r1su4zfuok0s7-8888.proxy.runpod.net/v1/models

# Test completion
curl -X POST https://4r1su4zfuok0s7-8888.proxy.runpod.net/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "prompt": "Say hello to Istanbul!",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

## Step 8: Update Your .env File

Once you confirm vLLM is working, update your `.env` file:

```bash
# Edit .env
nano .env

# Update these lines:
AI_ISTANBUL_LLM_MODE=runpod
PURE_LLM_MODE=true
LLM_API_URL=https://4r1su4zfuok0s7-8888.proxy.runpod.net

# Save: Ctrl+O, Enter, Ctrl+X
```

## Troubleshooting

### Issue 1: "Command not found: python"
Try:
```bash
python3 -m vllm.entrypoints.openai.api_server ...
```

### Issue 2: "No module named 'vllm'"
vLLM is not installed. Install it:
```bash
pip install vllm
```

### Issue 3: "CUDA out of memory"
Reduce GPU memory usage:
```bash
--gpu-memory-utilization 0.7  # Instead of 0.85
--max-model-len 1024          # Instead of 2048
```

### Issue 4: Model loading takes forever
- Check GPU is available: `nvidia-smi`
- Check disk space: `df -h`
- Check logs: `tail -f /workspace/vllm.log`

### Issue 5: Port 8888 already in use
Kill existing process:
```bash
# Find the process using port 8888
lsof -i :8888

# Kill it (replace PID with actual number)
kill -9 PID

# Or kill all vllm processes
pkill -f vllm
```

## Keep vLLM Running (Important!)

To prevent vLLM from stopping when you close SSH:

### Option A: Use Screen (Recommended)
```bash
# Install screen if needed
apt-get update && apt-get install -y screen

# Start screen session
screen -S vllm

# Inside screen, start vLLM (without nohup):
cd /workspace
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --port 8888 \
  --host 0.0.0.0

# Wait for "Application startup complete"
# Then detach from screen: Press Ctrl+A, then D

# To reconnect later:
screen -r vllm
```

### Option B: Use tmux
```bash
# Install tmux if needed
apt-get update && apt-get install -y tmux

# Start tmux session
tmux new -s vllm

# Inside tmux, start vLLM
cd /workspace
python -m vllm.entrypoints.openai.api_server ...

# Detach: Press Ctrl+B, then D
# Reconnect: tmux attach -t vllm
```

## Quick Reference Card

```bash
# ═══════════════════════════════════════════════════
# RUNPOD VLLM QUICK COMMANDS
# ═══════════════════════════════════════════════════

# 1. Check if running
ps aux | grep vllm

# 2. Start vLLM
cd /workspace
nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --port 8888 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &

# 3. Watch logs
tail -f /workspace/vllm.log

# 4. Test locally
curl http://localhost:8888/health

# 5. Test publicly (from Mac)
curl https://4r1su4zfuok0s7-8888.proxy.runpod.net/health

# 6. Stop vLLM
pkill -f vllm

# 7. View GPU status
nvidia-smi
```

## Next Steps

After vLLM is running:
1. ✅ Confirm public endpoint works
2. ✅ Update `.env` file with correct URL
3. ✅ Restart your backend (if running locally)
4. ✅ Test chat in frontend
5. ✅ Should see real AI responses, not fallback errors

Good luck! 🚀
