# 🎯 COMPLETE RUNPOD DEPLOYMENT GUIDE

## ⚠️ CRITICAL: Your SSH Public Key

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPn/II7Hndfgq1tkLKv0qMlZCBTdG9Nd4EovXG5hVxJE omertahtoko@gmail.com
```

**MUST DO FIRST:** Add this key to RunPod before proceeding!

---

## 📋 Complete Deployment Steps

### 1️⃣ Add SSH Key to RunPod

1. Go to: https://www.runpod.io/console/user/settings
2. Click "SSH Public Keys"
3. Click "Add SSH Key"
4. Paste the key above
5. Name: "Mac SSH Key"
6. Click "Add Key" ✅

### 2️⃣ Restart Your RunPod Instance

1. Go to: https://www.runpod.io/console/pods
2. Find pod: `pvj233wwhiu6j3-64411542`
3. Click "Stop" → Wait → Click "Start"
4. **Note the new SSH connection details!**

### 3️⃣ Test SSH Connection

Try both methods and see which works:

**Method A: Direct TCP**
```bash
# Get IP and port from RunPod dashboard
ssh -p <PORT> root@<IP_ADDRESS> -i ~/.ssh/id_ed25519 "echo 'Success!'"
```

**Method B: RunPod Proxy** (more reliable)
```bash
ssh pvj233wwhiu6j3-64411542@ssh.runpod.io -i ~/.ssh/id_ed25519 "echo 'Success!'"
```

### 4️⃣ Start vLLM on RunPod

SSH into RunPod (using whichever method worked above):

```bash
# Check if vLLM already running
ps aux | grep vllm

# Kill if needed
pkill -f vllm

# Start vLLM with optimized settings
python3 -m vllm.entrypoints.openai.api_server \
  --model /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 1024 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  > /root/vllm.log 2>&1 &

# Wait for startup
sleep 30

# Test locally
curl http://localhost:8000/v1/models
```

Expected: `{"object":"list","data":[{"id":"meta-llama/Llama-3.1-8B-Instruct"...`

### 5️⃣ Create SSH Tunnel (on your Mac)

Exit RunPod and run on your Mac:

```bash
# Kill old tunnels
pkill -f "ssh.*8000" 2>/dev/null; sleep 2

# Using RunPod Proxy (recommended):
ssh -f -N -L 8000:localhost:8000 \
  pvj233wwhiu6j3-64411542@ssh.runpod.io \
  -i ~/.ssh/id_ed25519 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3

# OR using Direct TCP (if that worked):
# ssh -f -N -L 8000:localhost:8000 root@<IP> -p <PORT> -i ~/.ssh/id_ed25519 ...

# Verify tunnel
ps aux | grep "ssh.*8000"
```

### 6️⃣ Test Tunnel

```bash
curl http://localhost:8000/v1/models
```

Expected: Same JSON response with model info

### 7️⃣ Start Backend

New terminal:

```bash
cd /Users/omer/Desktop/ai-stanbul/backend
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

### 8️⃣ Start Frontend

Another new terminal:

```bash
cd /Users/omer/Desktop/ai-stanbul/frontend
npm run dev
```

### 9️⃣ Test Locally

Open http://localhost:5173 and try: "Merhaba! Istanbul hakkında bilgi ver."

### 🔟 Expose with Ngrok

**Backend:**
```bash
ngrok http 5000
```
Copy the HTTPS URL (e.g., `https://abc123.ngrok-free.app`)

**Update Frontend:**
```bash
cd /Users/omer/Desktop/ai-stanbul/frontend
echo "VITE_API_BASE_URL=https://abc123.ngrok-free.app" > .env
```

Restart frontend (Ctrl+C then `npm run dev`)

**Frontend:**
```bash
ngrok http 5173
```

**Share the frontend URL!** 🎉

---

## 🔧 Troubleshooting

### "Permission denied (publickey)"
→ Did you add SSH key to RunPod? (Step 1)
→ Did you restart the pod? (Step 2)

### "Port not accessible" or "Connection refused"
→ Pod might be stopped or IP/port changed
→ Check RunPod dashboard for current connection info
→ Use RunPod proxy method instead

### vLLM not responding
```bash
# SSH into RunPod
ssh pvj233wwhiu6j3-64411542@ssh.runpod.io -i ~/.ssh/id_ed25519

# Check process
ps aux | grep vllm

# Check logs
tail -50 /root/vllm.log

# Check GPU
nvidia-smi
```

### Tunnel not working
```bash
# Kill and recreate
pkill -f "ssh.*8000"
# Then repeat Step 5
```

---

## 📚 Documentation

- `ADD_SSH_KEY_TO_RUNPOD.md` - Detailed SSH setup
- `DIRECT_TCP_DEPLOYMENT_GUIDE.md` - Alternative connection
- `RUNPOD_CONNECTION_TROUBLESHOOTING.md` - Detailed troubleshooting

---

## ✅ Success Checklist

- [ ] SSH key added to RunPod
- [ ] Pod restarted
- [ ] Can SSH into RunPod
- [ ] vLLM running on RunPod
- [ ] SSH tunnel created
- [ ] `curl localhost:8000/v1/models` works
- [ ] Backend started
- [ ] Frontend started
- [ ] Chatbot works locally
- [ ] Backend exposed via Ngrok
- [ ] Frontend configured and exposed
- [ ] Chatbot accessible publicly

---

**Start with Step 1 and work through each step in order!** 🚀
