# RunPod Endpoint Configuration Fix

## Problem
The URL you provided is the **terminal/SSH URL**, not the vLLM API endpoint:
```
❌ https://gbpd35labcq12f-19123.proxy.runpod.net/bx8f5opszjnt81kf2q5zw9tikfrvns23/
```

This URL returns an HTML page (terminal interface) instead of API responses.

## Solution

### Step 1: Find Your RunPod Pod Details
1. Go to https://www.runpod.io/console/pods
2. Click on your pod (the one running Llama 3.1 8B)
3. Look for the **"Connect"** button and the exposed ports

### Step 2: Identify the Correct Port
Your vLLM server should be running on one of these ports:
- Port **8888** (most common for vLLM)
- Port **8000** (alternative)
- Port **19123** (custom)

### Step 3: Construct the Correct API URL

RunPod proxy URLs follow this format:
```
https://{pod-id}-{port}.proxy.runpod.net/v1
```

Based on your pod ID `gbpd35labcq12f`, the correct URLs would be:

**Option 1: If vLLM is on port 8888**
```
https://gbpd35labcq12f-8888.proxy.runpod.net/v1
```

**Option 2: If vLLM is on port 8000**
```
https://gbpd35labcq12f-8000.proxy.runpod.net/v1
```

**Option 3: If vLLM is on port 19123**
```
https://gbpd35labcq12f-19123.proxy.runpod.net/v1
```

### Step 4: Test the Endpoint

Try each URL with this curl command:

```bash
# Test port 8888
curl -X POST https://gbpd35labcq12f-8888.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/llama-3.1-8b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'

# Test port 8000
curl -X POST https://gbpd35labcq12f-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/llama-3.1-8b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'

# Test port 19123
curl -X POST https://gbpd35labcq12f-19123.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/llama-3.1-8b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

You should get a JSON response like:
```json
{
  "id": "cmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "/workspace/llama-3.1-8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you?"
      },
      "finish_reason": "stop"
    }
  ]
}
```

### Step 5: Verify vLLM is Running on Your Pod

SSH into your RunPod and check:

```bash
# Check if vLLM process is running
ps aux | grep vllm

# Check what ports are listening
netstat -tuln | grep LISTEN

# Or check with lsof
lsof -i -P -n | grep LISTEN
```

### Step 6: If vLLM is Not Running

Start vLLM on port 8888:

```bash
cd /workspace
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/llama-3.1-8b \
  --quantization awq \
  --dtype float16 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8888 \
  > vllm.log 2>&1 &

# Check logs
tail -f vllm.log
```

### Step 7: Update Backend Configuration

Once you find the correct URL, update the backend:

1. **Update `.env` file:**
```bash
cd /Users/omer/Desktop/ai-stanbul
nano .env
```

Change:
```
LLM_API_URL=https://gbpd35labcq12f-XXXX.proxy.runpod.net/v1
```
(Replace XXXX with the correct port)

2. **Restart backend:**
```bash
cd backend
pkill -f main_pure_llm.py
python main_pure_llm.py > backend.log 2>&1 &
```

3. **Test:**
```bash
curl -X POST http://localhost:8002/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello from Istanbul!", "session_id": "test123"}'
```

## Common Issues

### Issue 1: Port Not Exposed
If the port isn't exposed, you need to:
1. Stop your pod
2. Edit pod settings
3. Add the port to "Exposed HTTP Ports"
4. Restart pod

### Issue 2: vLLM Not Started
Run the vLLM start command above.

### Issue 3: Wrong Model Name
Make sure the model name matches where it was downloaded:
```bash
ls -la /workspace/llama-3.1-8b
```

## Quick Checklist

- [ ] vLLM is running (check with `ps aux | grep vllm`)
- [ ] Port is exposed in RunPod settings
- [ ] Correct port in the proxy URL
- [ ] `/v1` is appended to the URL
- [ ] Backend `.env` is updated
- [ ] Backend is restarted
- [ ] Test endpoint returns JSON (not HTML)

## Need Help?

If you're still having issues, provide:
1. RunPod pod ID
2. Output of `ps aux | grep vllm`
3. Output of `netstat -tuln | grep LISTEN`
4. Output of testing the endpoint with curl
