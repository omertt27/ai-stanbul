# 🚀 START LLM SERVER NOW - Quick Start Guide

**Status:** ✅ All files ready, model downloaded (5.59 GB), server code tested  
**Goal:** Start the Llama 3.1 8B server on RunPod in under 5 minutes  
**Date:** January 2025

---

## 📍 YOUR RUNPOD CONNECTION DETAILS

```bash
# Direct TCP (Use this for SCP file uploads)
Host: 194.68.245.153
Port: 22077
User: root
SSH Key: ~/.ssh/id_ed25519

# RunPod Proxy (Alternative SSH method)
Host: 4r1su4zfuok0s7-64410d62@ssh.runpod.io
SSH Key: ~/.ssh/id_ed25519

# Web Terminal
Port: 19123
Access: RunPod Dashboard → Your Pod → "Open Web Terminal"
```

---

## 🎯 QUICK START (3 STEPS)

### Step 1: Upload Server Files (1 minute)

**From your local machine (`/Users/omer/Desktop/ai-stanbul`):**

```bash
# Upload LLM server
scp -P 22077 -i ~/.ssh/id_ed25519 llm_server.py root@194.68.245.153:/workspace/

# Upload startup script
scp -P 22077 -i ~/.ssh/id_ed25519 start_llm_server_runpod.sh root@194.68.245.153:/workspace/

# ✅ Both files uploaded successfully
```

### Step 2: SSH into RunPod (10 seconds)

```bash
ssh root@194.68.245.153 -p 22077 -i ~/.ssh/id_ed25519
```

**Alternative (Proxy method):**
```bash
ssh 4r1su4zfuok0s7-64410d62@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### Step 3: Start the Server (1 minute)

**On RunPod terminal:**

```bash
# Make script executable
chmod +x /workspace/start_llm_server_runpod.sh

# Start server
cd /workspace
./start_llm_server_runpod.sh
```

**Expected output:**
```
🚀 Starting LLM Server on RunPod...
✅ Installing dependencies...
✅ Creating logs directory...
✅ Starting server in background...
✅ Waiting for server to start (30s timeout)...
✅ Server is UP and healthy!
📝 Server logs: /workspace/logs/llm_server.log
🔍 Check logs: tail -f /workspace/logs/llm_server.log
```

---

## ✅ VERIFY SERVER IS RUNNING

### Test Health Endpoint

```bash
curl http://localhost:8001/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "device": "cuda",
  "memory_allocated_gb": 5.59
}
```

### Test Completion Endpoint

```bash
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is Istanbul famous for?",
    "max_tokens": 50
  }'
```

**Expected response:**
```json
{
  "id": "...",
  "object": "text_completion",
  "created": 1234567890,
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "choices": [{
    "text": "Istanbul is famous for its rich history...",
    "index": 0,
    "finish_reason": "length"
  }]
}
```

---

## 🔍 MONITORING

### View Server Logs

```bash
# Real-time logs
tail -f /workspace/logs/llm_server.log

# Last 50 lines
tail -n 50 /workspace/logs/llm_server.log

# Search for errors
grep -i error /workspace/logs/llm_server.log
```

### Check Server Process

```bash
# Check if server is running
ps aux | grep llm_server.py

# Check server PID
cat /workspace/llm_server.pid

# Check GPU usage
nvidia-smi

# Check memory usage
free -h
```

### Server Status

```bash
# Test health endpoint
curl http://localhost:8001/health

# Check if port is listening
netstat -tulpn | grep 8001
```

---

## 🛑 STOP/RESTART SERVER

### Stop Server

```bash
# Using PID file
kill $(cat /workspace/llm_server.pid)

# Or find and kill process
pkill -f llm_server.py

# Verify stopped
ps aux | grep llm_server.py
```

### Restart Server

```bash
# Stop first
kill $(cat /workspace/llm_server.pid)

# Wait 5 seconds
sleep 5

# Start again
cd /workspace
./start_llm_server_runpod.sh
```

---

## 🌐 EXPOSE SERVER (Optional - For Backend Connection)

### Option 1: Use RunPod Expose Feature

```bash
# In RunPod dashboard:
# 1. Go to your pod
# 2. Click "Expose" or "Edit"
# 3. Add HTTP port: 8001
# 4. Save changes
# 5. Note the public URL (e.g., https://xxxx.proxy.runpod.net:8001)
```

### Option 2: Use SSH Tunnel (Development)

**From your local machine:**

```bash
# Forward port 8001 from RunPod to local port 8001
ssh -L 8001:localhost:8001 root@194.68.245.153 -p 22077 -i ~/.ssh/id_ed25519

# Now access at: http://localhost:8001
```

### Option 3: Direct Access (If Pod Has Public IP)

```bash
# Test from local machine
curl http://194.68.245.153:8001/health

# If firewall blocks, configure in RunPod dashboard
```

---

## 🔗 CONNECT BACKEND TO LLM SERVER

### Update Backend Configuration

**File:** `backend/config/settings.py`

```python
# Add LLM server settings
LLM_SERVER_URL = "http://194.68.245.153:8001"  # Or tunneled/exposed URL
LLM_SERVER_API_KEY = ""  # Optional, if you add auth
LLM_TIMEOUT = 30  # seconds
LLM_MAX_RETRIES = 3
```

### Update Chat Endpoint

**File:** `backend/api/chat.py`

```python
import httpx

async def generate_with_llm(prompt: str, max_tokens: int = 150):
    """Generate response using RunPod LLM server"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.LLM_SERVER_URL}/v1/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["text"]
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return None
```

### Test Backend Integration

```bash
# Test from backend
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about Istanbul",
    "session_id": "test-123"
  }'
```

---

## 🐛 TROUBLESHOOTING

### Server Won't Start

```bash
# Check logs
tail -n 100 /workspace/logs/llm_server.log

# Check if port is in use
netstat -tulpn | grep 8001

# Kill existing process
pkill -f llm_server.py

# Try starting manually
cd /workspace
python llm_server.py
```

### Out of Memory

```bash
# Check GPU memory
nvidia-smi

# Restart server (clears memory)
kill $(cat /workspace/llm_server.pid)
./start_llm_server_runpod.sh

# If still OOM, reduce batch size in llm_server.py
```

### Model Not Found

```bash
# Verify model files
ls -lh /workspace/models/

# Re-download if needed
cd /workspace
./download_model.sh
```

### Connection Refused

```bash
# Check if server is running
ps aux | grep llm_server.py

# Check health endpoint
curl http://localhost:8001/health

# Check firewall (if accessing remotely)
# Configure in RunPod dashboard → Pod → Exposed Ports
```

---

## 📊 PERFORMANCE EXPECTATIONS

### Model Specifications
- **Model:** Llama 3.1 8B Instruct (4-bit quantized)
- **Size:** ~5.59 GB
- **GPU:** NVIDIA A40 (48 GB VRAM)
- **Quantization:** 4-bit NF4 with double quantization

### Expected Performance
- **Startup Time:** 30-60 seconds (model loading)
- **First Request:** 5-10 seconds (cold start)
- **Subsequent Requests:** 1-3 seconds
- **Tokens/Second:** 15-30 tokens/sec (depends on prompt length)
- **Max Tokens:** 128,256 (context window)
- **Concurrent Requests:** 1-3 (memory-limited)

### GPU Usage
- **Idle:** ~6 GB VRAM
- **During Inference:** ~8-12 GB VRAM
- **Temperature:** 60-80°C (normal)

---

## 📁 FILES ON RUNPOD

### After Setup, You Should Have:

```
/workspace/
├── llm_server.py              # FastAPI server (uploaded)
├── start_llm_server_runpod.sh # Startup script (uploaded)
├── llm_server.pid             # Process ID (auto-created)
├── models/                    # Model cache (already exists, 5.59 GB)
│   └── meta-llama/Meta-Llama-3.1-8B-Instruct/
└── logs/                      # Server logs (auto-created)
    └── llm_server.log
```

---

## ✅ SUCCESS CHECKLIST

- [ ] SSH key configured and working
- [ ] `llm_server.py` uploaded to `/workspace/`
- [ ] `start_llm_server_runpod.sh` uploaded to `/workspace/`
- [ ] Model files present in `/workspace/models/` (5.59 GB)
- [ ] Server started with `./start_llm_server_runpod.sh`
- [ ] Health endpoint returns `200 OK`
- [ ] Completion endpoint generates text
- [ ] Server logs show no errors
- [ ] GPU memory usage is ~6-8 GB
- [ ] Backend can connect to LLM server
- [ ] Test query returns coherent response

---

## 🎉 YOU'RE DONE!

Your LLM server is now running on RunPod and ready to power the Istanbul AI chatbot!

**Server URL:** `http://localhost:8001` (on RunPod)  
**Health Check:** `http://localhost:8001/health`  
**Completions:** `http://localhost:8001/v1/completions`  
**Chat:** `http://localhost:8001/v1/chat/completions`

### Next Steps:
1. ✅ Test all endpoints
2. ✅ Monitor logs and GPU usage
3. ✅ Connect backend to LLM server
4. ✅ Test end-to-end chatbot integration
5. ✅ Configure load balancing (if needed)
6. ✅ Set up monitoring/alerts

---

## 📚 RELATED DOCUMENTATION

- `RUNPOD_SETUP_COMPLETE_GUIDE.md` - Full setup guide with troubleshooting
- `LLM_SERVER_READY_STATUS.md` - Status report and file locations
- `LLM_SERVER_TEST_COMMANDS.md` - Detailed testing guide
- `PHASE4_3_4_CHATBOT_INTEGRATION_STATUS.md` - Phase 4.3/4.4 integration details
- `connect_runpod.sh` - Connection helper script

---

**Questions?** Check the troubleshooting section or logs: `/workspace/logs/llm_server.log`
