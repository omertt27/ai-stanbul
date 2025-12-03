# 🚀 RunPod LLM Server - Testing Steps

## Quick Reference
- **RunPod IP**: `194.68.245.153`
- **SSH Port**: `22077`
- **SSH Key**: `~/.ssh/id_ed25519`
- **User**: `root`
- **Server Port**: `8001`

---

## ⚡ STEP 1: Test SSH Connection

Run this command to verify you can connect to RunPod:

```bash
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 "echo '✅ SSH connection successful!'"
```

**Expected Output**: `✅ SSH connection successful!`

---

## 📦 STEP 2: Check if Model is Downloaded

Verify the Llama model exists on RunPod:

```bash
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 "ls -lh /workspace/models/"
```

**Expected Output**: Should show `meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf` (~5.59 GB)

**If model is missing**, download it:

```bash
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 "bash /workspace/download_model.sh"
```

---

## 🔧 STEP 3: Upload Server Scripts (if not already uploaded)

Upload the LLM server and startup scripts:

```bash
# Upload LLM server
scp -P 22077 -i ~/.ssh/id_ed25519 llm_server.py root@194.68.245.153:/workspace/

# Upload startup script
scp -P 22077 -i ~/.ssh/id_ed25519 start_llm_server_runpod.sh root@194.68.245.153:/workspace/

# Make startup script executable
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 "chmod +x /workspace/start_llm_server_runpod.sh"
```

---

## 🚀 STEP 4: Start the LLM Server

Start the server using the automated startup script:

```bash
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 "bash /workspace/start_llm_server_runpod.sh"
```

**Expected Output**:
```
═══════════════════════════════════════════════════════════════
🚀 STARTING LLM SERVER ON RUNPOD
═══════════════════════════════════════════════════════════════

✅ Found model: meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf

📂 Log directory: /workspace/logs
🗂️ PID file: /workspace/llm_server.pid

🚀 Starting LLM server...
✅ Server started! PID: 1234

⏳ Waiting for server to be ready...
✅ Server is healthy and responding!

🎉 LLM SERVER IS READY!
```

---

## 🧪 STEP 5: Run Automated Tests

Run the comprehensive test suite from your local machine:

```bash
./test_llm_server.sh
```

Or manually run each test:

### Test 1: Health Check
```bash
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'curl -s http://localhost:8001/health | python3 -m json.tool'
```

**Expected Output**:
```json
{
  "status": "healthy",
  "model": "meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf",
  "timestamp": "2025-12-03T..."
}
```

### Test 2: Text Completion
```bash
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 << 'ENDSSH'
curl -s -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Istanbul is",
    "max_tokens": 30,
    "temperature": 0.7
  }' | python3 -m json.tool
ENDSSH
```

**Expected Output**: JSON with generated text about Istanbul

### Test 3: GPU Memory Check
```bash
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader'
```

**Expected Output**: GPU memory usage (e.g., `5000 MiB, 24576 MiB, 50%`)

### Test 4: Server Process Check
```bash
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'ps aux | grep llm_server.py | grep -v grep'
```

**Expected Output**: Process details showing `python3 llm_server.py`

---

## 📊 STEP 6: Monitor Server Logs

### View live logs:
```bash
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'tail -f /workspace/logs/llm_server.log'
```

### View last 50 lines:
```bash
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'tail -50 /workspace/logs/llm_server.log'
```

### Search for errors:
```bash
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'grep -i error /workspace/logs/llm_server.log'
```

---

## 🔄 STEP 7: Stop/Restart Server (if needed)

### Stop the server:
```bash
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'kill $(cat /workspace/llm_server.pid) && rm /workspace/llm_server.pid'
```

### Check if server is stopped:
```bash
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'ps aux | grep llm_server.py | grep -v grep'
```

### Restart the server:
```bash
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 "bash /workspace/start_llm_server_runpod.sh"
```

---

## 🔌 STEP 8: Expose Port for External Access (Optional)

If you need to access the server from your backend or external network:

1. **Go to RunPod Dashboard**: https://www.runpod.io/console/pods
2. **Find your pod** and click on it
3. **Click "Edit"** or find the **Port Forwarding** section
4. **Add TCP Port**: `8001`
5. **Save changes**
6. **Note the public URL** provided by RunPod (e.g., `https://xxxx-8001.runpod.io`)

### Test external access:
```bash
curl https://xxxx-8001.runpod.io/health
```

---

## 🔗 STEP 9: Connect Backend to LLM Server

### Update your backend configuration:

**For internal RunPod access** (if backend is also on RunPod):
```python
LLM_SERVER_URL = "http://localhost:8001"
```

**For external access** (if backend is elsewhere):
```python
LLM_SERVER_URL = "https://xxxx-8001.runpod.io"  # Use the public URL from RunPod
```

### Example integration code:
```python
import requests

def get_llm_response(prompt: str, max_tokens: int = 100):
    """Get response from LLM server"""
    try:
        response = requests.post(
            f"{LLM_SERVER_URL}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

# Test it
result = get_llm_response("Istanbul is")
print(result)
```

---

## ✅ STEP 10: Verify Everything Works

### Final checklist:
- [ ] SSH connection works
- [ ] Model is downloaded (5.59 GB)
- [ ] Server starts successfully
- [ ] Health check returns `"status": "healthy"`
- [ ] Text completion generates Istanbul-related content
- [ ] GPU memory shows usage (~5-6 GB)
- [ ] Server process is running
- [ ] Logs show no errors
- [ ] (Optional) External port is exposed and accessible
- [ ] Backend can connect and get responses

---

## 🐛 Troubleshooting

### Server won't start:
```bash
# Check if port is already in use
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'lsof -i :8001'

# Check logs for errors
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'tail -100 /workspace/logs/llm_server.log'

# Check GPU availability
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'nvidia-smi'
```

### Out of GPU memory:
```bash
# Restart with lower context size
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'kill $(cat /workspace/llm_server.pid)'
# Edit llm_server.py and reduce n_ctx from 4096 to 2048
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 "bash /workspace/start_llm_server_runpod.sh"
```

### Server is slow:
```bash
# Check GPU utilization
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'nvidia-smi'

# Check system resources
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'htop'
```

---

## 📞 Quick Reference Commands

```bash
# Connect to RunPod
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153

# Start server
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 "bash /workspace/start_llm_server_runpod.sh"

# Test server
./test_llm_server.sh

# View logs
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'tail -f /workspace/logs/llm_server.log'

# Stop server
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'kill $(cat /workspace/llm_server.pid)'

# Check GPU
ssh -p 22077 -i ~/.ssh/id_ed25519 root@194.68.245.153 'nvidia-smi'
```

---

## 🎉 Success Indicators

Your server is **production-ready** when:

1. ✅ Health endpoint returns `200 OK`
2. ✅ Completions generate relevant text
3. ✅ GPU memory usage is stable (~5-6 GB)
4. ✅ No errors in logs
5. ✅ Response time < 5 seconds for 50 tokens
6. ✅ Server stays running for 10+ minutes
7. ✅ Backend successfully integrates and receives responses

---

**Ready to start? Run Step 1 and work through each step in order! 🚀**
