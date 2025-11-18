# RunPod Port 8000 Not Accessible - Solutions

## Problem
✅ vLLM is running on RunPod (confirmed via localhost:8000)
❌ External proxy URL returns 404

## Root Cause
RunPod's proxy isn't forwarding port 8000 externally, even though the service is running.

## Solutions (in order of preference)

### Solution 1: Check RunPod Console for HTTP Service URL
1. Go to: https://www.runpod.io/console/pods
2. Find your pod (ID: `fgkqzve33ssbea`)
3. Look for "Connect" button or "HTTP Services" section
4. Find the URL for **port 8000** specifically
5. Use that exact URL in your `.env` file

### Solution 2: Use RunPod's SSH Tunnel
If port 8000 isn't exposed as HTTP, you can tunnel it:

On your Mac:
```bash
# Get your RunPod SSH connection string from the console
ssh -L 8000:localhost:8000 root@ssh.runpod.io -i ~/.ssh/your-runpod-key

# Then use this in .env:
LLM_API_URL=http://localhost:8000/v1
```

### Solution 3: Change vLLM to Use a Different Port
Pick a port that IS exposed (like 8888 or create a new one):

**On RunPod terminal:**
```bash
# Stop current vLLM (Ctrl+C)

# Start on port 8888 instead
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8888
```

**Then update .env:**
```properties
LLM_API_URL=https://fgkqzve33ssbea-8888.proxy.runpod.net/v1
```

### Solution 4: Add Port 8000 to RunPod Configuration
1. Stop your pod
2. Edit pod configuration
3. Add port 8000 to "TCP Port Mappings" or "Exposed Ports"
4. Start pod again
5. Restart vLLM

## Quick Test Commands

**Inside RunPod (these should work):**
```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/health
```

**From your Mac (test external access):**
```bash
curl https://fgkqzve33ssbea-8000.proxy.runpod.net/v1/models
```

## Recommended Next Step
**Try Solution 3** (use port 8888) - it's the quickest and you know port 8888 is already exposed!
