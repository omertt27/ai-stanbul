# How to Start vLLM on RunPod - Complete Guide

## 🚨 CRITICAL ISSUE
**Port 8000 is NOT exposed on your RunPod pod!**

Your pod currently has:
- ✅ Port 8888 (Jupyter Lab) - Working
- ✅ Port 22 (SSH) - Working  
- ❌ **Port 8000 (vLLM API) - NOT EXPOSED**

## Problem
You need to:
1. **Start vLLM** on port 8000 (inside the pod)
2. **Expose port 8000** via RunPod UI (so it's accessible externally)

## Solution: Start vLLM Server

### Method 1: Via Jupyter Lab Terminal (EASIEST)

1. **Open Jupyter Lab**: https://pvj233wwhiu6j3-8888.proxy.runpod.net

2. **Open Terminal**: File → New → Terminal

3. **Check if vLLM is already running**:
   ```bash
   ps aux | grep vllm
   ```
   
   If you see a process, kill it first:
   ```bash
   pkill -9 -f vllm
   ```

4. **Start vLLM Server**:
   ```bash
   python3 -m vllm.entrypoints.openai.api_server \
     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
     --port 8000 \
     --host 0.0.0.0 \
     --dtype auto \
     --max-model-len 4096 \
     --gpu-memory-utilization 0.9 \
     > /root/vllm.log 2>&1 &
   ```

5. **Verify it's running**:
   ```bash
   sleep 10
   curl http://localhost:8000/health
   ```
   
   Should return: `OK` or similar

6. **Check the logs**:
   ```bash
   tail -f /root/vllm.log
   ```
   
   Wait until you see: `"Application startup complete"`

---

### Method 2: Expose Port 8000 in RunPod UI

**After starting vLLM**, you MUST expose port 8000:

1. **Go to RunPod Console**: https://www.runpod.io/console/pods

2. **Find your pod** (`pvj233wwhiu6j3` / modest_azure_mackerel)

3. **Click on the pod** to open details

4. **Find "HTTP Services" section**

5. **Click "+ Add Port" or "Edit"**

6. **Add new service**:
   - **Port:** `8000`
   - **Protocol:** `HTTP`
   - **Name:** `vLLM API` (optional)

7. **Save**

8. **Verify** you see:
   ```
   Port 8000
   vLLM API  
   https://pvj233wwhiu6j3-8000.proxy.runpod.net
   ```

---

### Method 3: Via SSH

```bash
ssh root@pvj233wwhiu6j3-64411542.ssh.runpod.io

# Check if running
ps aux | grep vllm

# Kill if needed
pkill -9 -f vllm

# Start vLLM
python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  > /root/vllm.log 2>&1 &

# Wait and verify
sleep 10
curl http://localhost:8000/health
```

### Method 4: Create Startup Script (PERSISTENT)

Create a script that starts vLLM automatically on pod restart.

1. **Open Jupyter Lab terminal**

2. **Create startup script**:
   ```bash
   cat > /root/start_vllm.sh << 'EOF'
#!/bin/bash
echo "Starting vLLM server..."

# Kill any existing vLLM processes
pkill -9 -f vllm

# Wait a moment
sleep 2

# Start vLLM with optimal settings for RTX A5000
python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  > /root/vllm.log 2>&1 &

echo "vLLM server started in background"
echo "Check logs: tail -f /root/vllm.log"
echo "Check health: curl http://localhost:8000/health"
EOF
   ```

3. **Make it executable**:
   ```bash
   chmod +x /root/start_vllm.sh
   ```

4. **Run it**:
   ```bash
   /root/start_vllm.sh
   ```

5. **Add to startup** (optional - runs on pod restart):
   ```bash
   echo "/root/start_vllm.sh" >> /root/.bashrc
   ```

## Verification Steps

After starting vLLM, verify it's working:

### 1. Check Process
```bash
ps aux | grep vllm
```

Should show a Python process with vLLM.

### 2. Check Logs
```bash
tail -50 /root/vllm.log
```

Look for:
- ✅ `Application startup complete`
- ✅ `Uvicorn running on http://0.0.0.0:8000`
- ❌ `CUDA out of memory` (if you see this, reduce max-model-len)
- ❌ `Model not found` (check model name)

### 3. Test Health Endpoint (Local)
```bash
curl http://localhost:8000/health
```

Expected: `OK` or `{"status":"ok"}`

### 4. Test Models Endpoint (Local)
```bash
curl http://localhost:8000/v1/models
```

Expected: JSON with model list

### 5. Test Generation (Local)
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Hello, what is Istanbul?",
    "max_tokens": 50
  }'
```

Expected: JSON response with generated text

### 6. Test External Access
```bash
# From your local machine
curl https://pvj233wwhiu6j3-8000.proxy.runpod.net/v1/models
```

Expected: Same model list as step 4

## Common Issues & Solutions

### Issue: Model Not Found
**Error**: `Model meta-llama/Meta-Llama-3.1-8B-Instruct not found`

**Solution 1**: Check available models
```bash
ls -la /workspace/
ls -la /root/.cache/huggingface/hub/
```

**Solution 2**: Download the model
```bash
pip install huggingface-hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Meta-Llama-3.1-8B-Instruct')"
```

**Solution 3**: Use local path if model is already downloaded
```bash
# Find the model
find /workspace -name "*llama*" -type d 2>/dev/null
find /root/.cache -name "*llama*" -type d 2>/dev/null

# Use the full path
python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --port 8000 \
  --host 0.0.0.0
```

### Issue: CUDA Out of Memory
**Error**: `torch.cuda.OutOfMemoryError`

**Solution**: Reduce memory usage
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype auto \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 1
```

### Issue: Port Already in Use
**Error**: `Address already in use`

**Solution**: Kill existing process
```bash
# Find process on port 8000
lsof -i :8000

# Kill it
pkill -9 -f vllm

# Or kill by PID
kill -9 <PID>
```

### Issue: vLLM Not Installed
**Error**: `No module named 'vllm'`

**Solution**: Install vLLM
```bash
pip install vllm
# Or if you need specific version
pip install vllm==0.2.7
```

### Issue: Permission Denied
**Error**: `Permission denied: '/root/vllm.log'`

**Solution**: Use different log path
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  > /tmp/vllm.log 2>&1 &
```

## Alternative: Using Docker (Advanced)

If vLLM is giving issues, you can run it in a container:

```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  --name vllm-server \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

## Next Steps After Starting vLLM

1. **Verify external access**:
   ```bash
   # From your local machine
   curl https://pvj233wwhiu6j3-8000.proxy.runpod.net/v1/models
   ```

2. **Run test script**:
   ```bash
   cd /Users/omer/Desktop/ai-stanbul
   python test_runpod_connection.py
   ```

3. **Start backend**:
   ```bash
   cd /Users/omer/Desktop/ai-stanbul/backend
   source venv/bin/activate
   uvicorn main:app --reload --port 8010
   ```

4. **Start frontend**:
   ```bash
   cd /Users/omer/Desktop/ai-stanbul/frontend
   npm run dev
   ```

5. **Test chat**: http://localhost:5173

## Success Indicators

✅ `ps aux | grep vllm` shows running process  
✅ `curl localhost:8000/health` returns OK  
✅ `curl localhost:8000/v1/models` returns model list  
✅ External URL accessible from your machine  
✅ Test script passes all tests  
✅ Backend connects successfully  
✅ Frontend chat receives responses  

## Quick Reference Commands

```bash
# Start vLLM
python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 --host 0.0.0.0 \
  > /root/vllm.log 2>&1 &

# Check if running
ps aux | grep vllm

# Check logs
tail -f /root/vllm.log

# Test locally
curl http://localhost:8000/health

# Kill if needed
pkill -9 -f vllm

# Restart
pkill -9 -f vllm && sleep 2 && \
python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 --host 0.0.0.0 \
  > /root/vllm.log 2>&1 &
```

---

## Related Documentation
- `RUNPOD_PORT_8000_EXPOSURE.md` - How to expose port 8000
- `FINAL_DEPLOYMENT_CHECKLIST.md` - Complete deployment steps
- `LLM_FIX_GUIDE.md` - LLM integration guide
