# 🔧 RunPod Connection Troubleshooting

## Issue: Port 22048 Not Accessible

### Possible Causes:
1. RunPod pod is stopped or terminated
2. Port mapping has changed
3. RunPod assigned a different SSH port
4. Firewall blocking the connection

## 🔍 Diagnosis Steps

### 1. Check RunPod Pod Status

Log into RunPod dashboard: https://www.runpod.io/console/pods

Check:
- Is the pod running? (green status)
- What is the current SSH port?
- Is the pod ID still: `pvj233wwhiu6j3-64411542`

### 2. Find Current SSH Connection Details

In the RunPod pod page, look for:
- **SSH Connection Command** (usually shows: `ssh root@<IP> -p <PORT>`)
- Example: `ssh root@161.97.138.99 -p 22048`

The port might have changed if the pod was restarted!

### 3. Try RunPod Proxy SSH (Alternative)

If direct TCP doesn't work, try the RunPod proxy:

```bash
# Format: ssh <POD_ID>-<INTERNAL_ID>@ssh.runpod.io
ssh pvj233wwhiu6j3-64411542@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### 4. Test Connection Methods

**Method A: Direct TCP (Preferred)**
```bash
# Replace PORT with current SSH port from RunPod dashboard
ssh -p 22048 root@161.97.138.99 -i ~/.ssh/id_ed25519
```

**Method B: RunPod Proxy**
```bash
ssh pvj233wwhiu6j3-64411542@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### 5. Update Connection Details

Once you find the correct connection method, update these files:

**setup_direct_tcp_tunnel.sh:**
```bash
# Update these variables
RUNPOD_IP="161.97.138.99"      # Update if changed
RUNPOD_PORT="22048"             # Update if changed
```

**Or use RunPod proxy in setup_fresh_tunnel.sh:**
```bash
ssh -f -N -L 8000:localhost:8000 \
  pvj233wwhiu6j3-64411542@ssh.runpod.io \
  -i ~/.ssh/id_ed25519 \
  # ... rest of options
```

## 🚀 Quick Fixes

### Option 1: Use RunPod Proxy SSH

If direct TCP isn't working, use the RunPod proxy:

```bash
# Kill old tunnels
pkill -f "ssh.*runpod" 2>/dev/null || true

# Create tunnel via RunPod proxy
ssh -f -N -L 8000:localhost:8000 \
  pvj233wwhiu6j3-64411542@ssh.runpod.io \
  -i ~/.ssh/id_ed25519 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -o TCPKeepAlive=yes

# Test
curl http://localhost:8000/v1/models
```

### Option 2: Check if Pod is Stopped

If the pod is stopped in RunPod dashboard:
1. Click "Start" to restart the pod
2. Wait for it to fully start (green status)
3. Note the new SSH connection details
4. Re-run vLLM startup script

### Option 3: Restart vLLM on RunPod

SSH into RunPod (using whichever method works) and restart vLLM:

```bash
# Kill existing vLLM
pkill -f vllm

# Start vLLM
python3 -m vllm.entrypoints.openai.api_server \
  --model /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 1024 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  > /root/vllm.log 2>&1 &

# Verify
curl http://localhost:8000/v1/models
```

## 📋 Connection Matrix

| Method | Connection String | Pros | Cons |
|--------|------------------|------|------|
| Direct TCP | `root@161.97.138.99 -p 22048` | Faster, more stable | Port may change on restart |
| RunPod Proxy | `pvj233wwhiu6j3-64411542@ssh.runpod.io` | Always works | Slightly slower, extra hop |

## ✅ Success Checklist

After fixing the connection:

```bash
# 1. Test SSH connection
ssh <CONNECTION_STRING> -i ~/.ssh/id_ed25519 "echo 'Connection works'"

# 2. Check vLLM is running
ssh <CONNECTION_STRING> -i ~/.ssh/id_ed25519 "ps aux | grep vllm"

# 3. Test vLLM locally on RunPod
ssh <CONNECTION_STRING> -i ~/.ssh/id_ed25519 "curl http://localhost:8000/v1/models"

# 4. Create SSH tunnel
ssh -f -N -L 8000:localhost:8000 <CONNECTION_STRING> -i ~/.ssh/id_ed25519

# 5. Test tunnel
curl http://localhost:8000/v1/models
```

## 🆘 Still Not Working?

### Check These:

1. **RunPod Pod Status:** Is it actually running?
2. **vLLM Process:** Is it running on the pod?
3. **SSH Key:** Is it added to RunPod?
4. **Firewall:** Is your Mac blocking outbound connections?
5. **RunPod Credits:** Do you have credits remaining?

### Get Help:

1. Check RunPod logs in dashboard
2. SSH into pod and check `/root/vllm.log`
3. Run: `nvidia-smi` to check GPU status
4. Run: `df -h` to check disk space

## 📞 Emergency Fallback

If nothing works, you can temporarily use OpenAI API while debugging:

1. Edit `backend/.env`:
   ```
   LLM_API_URL=https://api.openai.com/v1
   LLM_MODEL_NAME=gpt-3.5-turbo
   OPENAI_API_KEY=<your-key>
   ```

2. Restart backend

3. Chatbot will work with OpenAI while you debug RunPod

---

**Remember:** RunPod pods can be stopped to save credits. Always check the pod status first!
