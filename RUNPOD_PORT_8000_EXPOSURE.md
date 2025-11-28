# RunPod Port 8000 Exposure Guide

## Current Status
- ✅ Pod ID: `pvj233wwhiu6j3`
- ✅ **vLLM server RUNNING on port 8000** (internal) 🎉
- ✅ Model loaded: Meta-Llama-3.1-8B-Instruct
- ✅ Server accessible locally inside pod
- ⚠️ **External proxy not routing yet** - Use SSH tunnel workaround
- ✅ Port 8888 exposed (Jupyter Lab): https://pvj233wwhiu6j3-8888.proxy.runpod.net
- ✅ Port 22 exposed (SSH): ssh.runpod.io

## ✅ vLLM is Running!

The LLM server is fully operational inside the pod! Since RunPod's HTTP proxy isn't routing to port 8000 yet, use the SSH tunnel method below.

**Quick Fix: SSH Tunnel (Recommended)**

## Goal
Expose port 8000 so the vLLM API is accessible at:
```
https://pvj233wwhiu6j3-8000.proxy.runpod.net/v1
```

---

## Method 1: RunPod Web UI (RECOMMENDED)

### Steps:
1. **Go to RunPod Dashboard**: https://www.runpod.io/console/pods
2. **Find your pod** with ID `pvj233wwhiu6j3`
3. **Click on the pod** to open details
4. **Look for "HTTP Services" or "Exposed Ports" section**
5. **Click "Edit" or "+ Add Port"**
6. **Add the following**:
   - Port: `8000`
   - Protocol: `HTTP`
   - Name: `vLLM API` (optional)
7. **Save changes**
8. **Verify** the new URL appears in the list

### Expected Result:
After saving, you should see:
```
Port 8000
vLLM API
https://pvj233wwhiu6j3-8000.proxy.runpod.net
```

---

## Method 2: Using Pod's Terminal (Jupyter Lab)

### Quick Start (Copy-Paste This!):

1. **Open Jupyter Lab**: https://pvj233wwhiu6j3-8888.proxy.runpod.net
2. **Open a Terminal** (File → New → Terminal)
3. **Copy-paste this entire block**:

```bash
# Kill any existing vLLM
pkill -9 -f vllm

# Start vLLM server
python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  > /root/vllm.log 2>&1 &

# Wait 30 seconds for startup
sleep 30

# Test it
curl http://localhost:8000/health
```

4. **Wait for "Application startup complete"** in logs:
   ```bash
   tail -f /root/vllm.log
   ```
   Press Ctrl+C to exit log viewing

5. **Verify it works**:
   ```bash
   curl http://localhost:8000/v1/models
   ```

### Detailed Instructions:

3. **Verify vLLM is running**:
   ```bash
   curl http://localhost:8000/health
   ```
   Should return: `{"status":"ok"}`

4. **Check vLLM process**:
   ```bash
   ps aux | grep vllm
   ```

5. **If vLLM is NOT running**, start it:
   ```bash
   python3 -m vllm.entrypoints.openai.api_server \
     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
     --port 8000 \
     --host 0.0.0.0 \
     --dtype auto \
     --max-model-len 4096
   ```

**For complete troubleshooting, see:** `START_VLLM_ON_RUNPOD.md`

---

## Method 3: SSH Tunnel (Local Testing)

If you want to test locally before exposing publicly:

1. **Run the tunnel script**:
   ```bash
   ./expose_port_runpod.sh
   ```

2. **Test locally**:
   ```bash
   curl http://localhost:8000/health
   ```

3. **Test chat completion**:
   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 50
     }'
   ```

---

## Method 4: RunPodCTL (Advanced)

If you have `runpodctl` installed:

1. **Install runpodctl** (if not installed):
   ```bash
   brew install runpod/runpodctl/runpodctl
   ```

2. **Configure**:
   ```bash
   runpodctl config
   ```

3. **List pods**:
   ```bash
   runpodctl get pods
   ```

4. **Expose port**:
   ```bash
   runpodctl expose pvj233wwhiu6j3 8000
   ```

---

## Verification Steps

After exposing port 8000, verify it works:

### 1. Test Health Endpoint
```bash
curl https://pvj233wwhiu6j3-8000.proxy.runpod.net/health
```

Expected response:
```json
{"status":"ok"}
```

### 2. Test Models Endpoint
```bash
curl https://pvj233wwhiu6j3-8000.proxy.runpod.net/v1/models
```

Expected response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
      "object": "model",
      ...
    }
  ]
}
```

### 3. Test Chat Completion
```bash
curl https://pvj233wwhiu6j3-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Merhaba Istanbul!"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### 4. Run Python Test Script
```bash
cd /Users/omer/Desktop/ai-stanbul
python test_runpod_connection.py
```

---

## Troubleshooting

### Port 8000 doesn't appear in RunPod UI
- **Solution**: Port exposure might be a pro feature. Use SSH tunnel instead.
- **Alternative**: Contact RunPod support to enable port exposure.

### vLLM not responding
1. **Check if running**:
   ```bash
   ssh pvj233wwhiu6j3-64411542@ssh.runpod.io "ps aux | grep vllm"
   ```

2. **Check logs**:
   ```bash
   ssh pvj233wwhiu6j3-64411542@ssh.runpod.io "tail -100 /root/vllm.log"
   ```

3. **Restart vLLM**:
   - Open Jupyter Lab terminal
   - Kill existing process: `pkill -f vllm`
   - Start new one (see Method 2, step 5)

### Connection timeout
- Verify firewall rules on RunPod
- Check if pod is still running
- Verify URL format: `https://PODID-PORT.proxy.runpod.net`

### 404 Not Found
- Add `/v1` to the end of base URL
- Example: `https://pvj233wwhiu6j3-8000.proxy.runpod.net/v1/chat/completions`

---

## Next Steps

After successfully exposing port 8000:

1. ✅ **Update backend/.env**:
   ```
   LLM_API_URL=https://pvj233wwhiu6j3-8000.proxy.runpod.net/v1
   ```

2. ✅ **Restart backend**:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

3. ✅ **Start frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

4. ✅ **Test chat functionality**:
   - Open http://localhost:5173
   - Send a test message
   - Verify Llama 3.1 8B responds

5. ✅ **Check About page**:
   - Verify it mentions "Llama 3.1 8B"
   - Verify all translations are correct

---

## Success Criteria

✅ Port 8000 visible in RunPod HTTP Services  
✅ `curl https://pvj233wwhiu6j3-8000.proxy.runpod.net/health` returns `{"status":"ok"}`  
✅ `curl https://pvj233wwhiu6j3-8000.proxy.runpod.net/v1/models` returns model list  
✅ Chat completion test returns valid response  
✅ Backend connects successfully to vLLM  
✅ Frontend chat sends messages and receives responses  
✅ About page displays "Llama 3.1 8B"  

---

## References
- RunPod Documentation: https://docs.runpod.io/
- vLLM Documentation: https://docs.vllm.ai/
- Related guides: `FINAL_DEPLOYMENT_CHECKLIST.md`, `RUNPOD_PORT_EXPOSURE_GUIDE.md`
