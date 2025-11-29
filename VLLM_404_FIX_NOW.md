# CRITICAL FIX: vLLM 404 Error on Render

## Issue Found! 🎯

**From Render logs**:
```
HTTP Request: POST https://i6c58scsmccj2s-8888.proxy.runpod.net/v1/completions "HTTP/1.1 404 Not Found"
❌ LLM HTTP error: 404
❌ LLM generation failed: Invalid LLM response
```

**Good news**: 
- ✅ Backend IS configured correctly (`pure_llm_core exists: True`)
- ✅ Environment variables ARE set (`LLM client exists: True`)
- ✅ Backend code IS deployed

**Bad news**:
- ❌ vLLM endpoint returns 404 when Render calls it
- ❌ This causes fallback error response

## Root Cause Analysis

The 404 error can happen for several reasons:

### Reason 1: vLLM Server Not Running (Most Likely)
RunPod pods can go to sleep or be terminated if:
- Pod is idle for too long
- Manual stop
- Out of credits
- Pod crashed

### Reason 2: vLLM Endpoint Changed
- Pod restarted with different port
- Proxy URL changed
- Need to update endpoint URL

### Reason 3: Network/Firewall Issue
- RunPod might block requests from Render's IPs
- Rate limiting
- Temporary network issue

## IMMEDIATE FIX: Restart vLLM on RunPod

### Step 1: Check Pod Status

1. Go to: https://www.runpod.io/console/pods
2. Find your pod: `i6c58scsmccj2s` (or `nnqisfv2zk46t2`)
3. Check status:
   - ✅ **Running** (green) = Good
   - ⚠️ **Stopped** (red) = Need to start
   - 💤 **Sleeping** (yellow) = Need to wake up

### Step 2: If Pod is Stopped - Start It

1. Click on the pod
2. Click **"Start"** button
3. Wait 1-2 minutes for pod to boot

### Step 3: SSH into Pod and Restart vLLM

```bash
# SSH into your RunPod pod
ssh root@your-pod-id-ssh.proxy.runpod.net -p your-ssh-port

# Check if vLLM is running
ps aux | grep vllm

# If NOT running, start it:
cd /workspace

# Start vLLM in background
nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --port 8888 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &

# Check logs
tail -f /workspace/vllm.log

# Wait for: "Application startup complete"
# Then press Ctrl+C to exit tail
```

### Step 4: Verify vLLM is Running

```bash
# Test locally on pod
curl http://localhost:8888/health

# Test public endpoint from your Mac
curl https://i6c58scsmccj2s-8888.proxy.runpod.net/v1/models
```

Expected: JSON response with model info

### Step 5: Test Backend Again

```bash
# From your Mac
./test_render_backend.sh
```

Expected: Real LLM responses, not fallback

## Alternative Solution: Use Screen/Systemd

To prevent vLLM from stopping when SSH disconnects:

### Option A: Use Screen (Simple)

```bash
# In RunPod SSH:
screen -S vllm

# Inside screen, start vLLM:
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --port 8888 \
  --host 0.0.0.0

# Wait for "Application startup complete"
# Then detach: Press Ctrl+A, then D

# To reattach later: screen -r vllm
```

### Option B: Create a Startup Script

```bash
# In RunPod SSH:
cat > /workspace/start_vllm.sh << 'EOF'
#!/bin/bash
cd /workspace
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --port 8888 \
  --host 0.0.0.0
EOF

chmod +x /workspace/start_vllm.sh

# Run it
/workspace/start_vllm.sh > /workspace/vllm.log 2>&1 &
```

## Long-Term Solutions

### Option 1: Keep RunPod Pod Always On
- Set pod to "Always On" (costs more)
- Or check "Keep pod running" in settings

### Option 2: Add Auto-Restart to RunPod
Create a startup script that runs on pod boot:
```bash
# In /workspace/.profile or similar
/workspace/start_vllm.sh &
```

### Option 3: Move to More Stable Hosting
- **Render**: Deploy vLLM as a separate service (more expensive but more reliable)
- **Hugging Face Inference API**: Managed service, no maintenance
- **Modal.com**: Serverless GPU compute
- **Replicate.com**: Managed model hosting

### Option 4: Add Fallback LLM
Update backend to use backup LLM if RunPod fails:

```python
# In runpod_llm_client.py
FALLBACK_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct"

async def generate_with_fallback(prompt):
    try:
        return await generate_runpod(prompt)
    except:
        logger.warning("RunPod failed, using Hugging Face fallback")
        return await generate_huggingface(prompt)
```

## Quick Checklist

- [ ] Check RunPod pod status (running?)
- [ ] SSH into pod
- [ ] Check if vLLM process is running (`ps aux | grep vllm`)
- [ ] If not, start vLLM with command above
- [ ] Test endpoint: `curl localhost:8888/health`
- [ ] Test from Mac: `curl https://i6c58scsmccj2s-8888.proxy.runpod.net/v1/models`
- [ ] Run `./test_render_backend.sh`
- [ ] Test frontend chat

## Monitor vLLM Status

Create a simple health check endpoint:

```bash
# Add to backend /api/debug/llm-status
@router.get("/debug/llm-status")
async def llm_status():
    from services.runpod_llm_client import get_llm_client
    client = get_llm_client()
    
    if not client or not client.enabled:
        return {"status": "disabled"}
    
    health = await client.health_check()
    return health
```

Then monitor: `https://ai-stanbul.onrender.com/api/debug/llm-status`

## Expected Timeline

- **Check pod status**: 1 minute
- **SSH into pod**: 1 minute  
- **Start vLLM**: 2-3 minutes (model loading)
- **Test endpoint**: 1 minute
- **Verify backend**: 2 minutes

**Total**: ~7-8 minutes

## What Happens Next

Once vLLM is running:
1. ✅ Backend will connect successfully
2. ✅ Chat will return real LLM responses
3. ✅ No more fallback errors
4. ✅ Response time will be fast (~3-5 seconds)

## Prevention

To prevent this from happening again:

1. **Keep pod running**: Check RunPod settings
2. **Monitor pod status**: Set up alerts
3. **Auto-restart**: Add startup script
4. **Fallback LLM**: Implement backup service
5. **Better hosting**: Consider moving to managed service

## Files Referenced

- `/RUNPOD_VLLM_SETUP.md` - Original setup guide
- `/START_VLLM_4BIT_OPTIMIZED.md` - Optimized startup command
- `/test_render_backend.sh` - Test script
- `/backend/services/runpod_llm_client.py` - LLM client code

## Summary

**Problem**: vLLM returned 404 → Backend fallback  
**Cause**: vLLM not running on RunPod  
**Fix**: Restart vLLM service on pod  
**Time**: ~7 minutes  
**Prevention**: Keep pod running or add auto-restart  

Now go restart that vLLM server! 🚀
