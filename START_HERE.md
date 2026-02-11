# 🚀 START HERE - Istanbul AI Chatbot Deployment

## Your Mission: Deploy Llama 3.1 8B Chatbot to the Internet

### Current Status: 99% Complete! 🎉

```
✅ Model downloaded (Llama 3.1 8B)
✅ vLLM configured
✅ Backend ready
✅ Frontend ready
⚠️  Missing: SSH key on RunPod
```

**⚠️ CRITICAL: Run all commands on your Mac, NOT inside Docker/containers!**

---

## 🔑 YOUR SSH KEY (Copy This!)

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPn/II7Hndfgq1tkLKv0qMlZCBTdG9Nd4EovXG5hVxJE omertahtoko@gmail.com
```

**⚠️ IMPORTANT:** This key MUST be added to RunPod before anything will work!

---

## ❓ Getting "Permission denied (publickey)"?

This means the SSH key isn't added to RunPod yet. You MUST:
1. Add the SSH key above to RunPod settings
2. **RESTART the pod** (critical - key won't work until restart!)
3. Wait for pod to fully start (green status)
4. Then try SSH again

---

## 🎯 10-Minute Deployment Plan

### ⏱️ Minute 1-2: Add SSH Key

1. Open: https://www.runpod.io/console/user/settings
2. Click "SSH Public Keys"
3. Click "Add SSH Key"
4. Paste the key above
5. Name: "Mac SSH Key"
6. Save ✅

### ⏱️ Minute 3-4: Restart Pod

1. Go to: https://www.runpod.io/console/pods
2. Find: `p9lf44wbmqzs5l-64411547`
3. Stop → Wait → Start
4. Note SSH connection details ✅

### ⏱️ Minute 5: Test SSH

**⚠️ IMPORTANT: Run these commands on your MAC, not inside Docker/containers!**

Open a **new terminal on your Mac** and run:

```bash
# Method 1: RunPod Proxy (Recommended - works behind firewalls)
ssh p9lf44wbmqzs5l-64411547@ssh.runpod.io -i ~/.ssh/id_ed25519

# Method 2: Direct TCP (Faster, supports SCP/SFTP)
ssh root@69.30.85.235 -p 22100 -i ~/.ssh/id_ed25519

# Quick test (option 1)
ssh p9lf44wbmqzs5l-64411547@ssh.runpod.io -i ~/.ssh/id_ed25519 echo OK
```

**Expected:** You should be logged into RunPod!

**If you get "Permission denied (publickey)":**
- Go back to Step 1-2 and add the SSH key
- Make sure you restart the pod in Step 3-4
- The key won't work until the pod is restarted!

**If you get "Identity file not accessible":**
- You're in the wrong terminal (Docker/container)
- Open a **new Mac terminal** (not inside any container)
- The SSH key is at `/Users/omer/.ssh/id_ed25519` on your Mac

If you see the RunPod shell prompt, continue! ✅

### ⏱️ Minute 6-7: Start vLLM (If Not Already Running)

First, check if vLLM is already running:
```bash
# SSH into RunPod
ssh p9lf44wbmqzs5l-64411547@ssh.runpod.io -i ~/.ssh/id_ed25519

# Check if vLLM is running
curl http://localhost:8000/v1/models
```

**If it works:** Skip to Minute 8! ✅

**If it fails:** Start vLLM:
```bash
# Find your model path
ls -la /workspace/models/  # Check if model is here
# OR
ls -la /root/.cache/huggingface/hub/

# Start vLLM (adjust model path if needed)
python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000 --dtype float16 --max-model-len 1024 \
  --gpu-memory-utilization 0.85 > /workspace/vllm.log 2>&1 &
```

Wait 30 seconds, then test:
```bash
curl http://localhost:8000/v1/models
```

See model info? Continue! ✅

### ⏱️ Minute 8: Create Tunnel

On your Mac (in a new terminal):

```bash
# Method 1: RunPod Proxy (Recommended)
ssh -f -N -L 8000:localhost:8000 p9lf44wbmqzs5l-64411547@ssh.runpod.io -i ~/.ssh/id_ed25519

# Method 2: Direct TCP (Alternative - faster)
ssh -f -N -L 8000:localhost:8000 root@69.30.85.235 -p 22100 -i ~/.ssh/id_ed25519

# Test the tunnel
curl http://localhost:8000/v1/models
```

**Expected output:** Should show Llama-3.1-8B-Instruct model info

See model info? Continue! ✅

**Troubleshooting:**
- If tunnel command gives error: Remove `-f` flag and run in separate terminal
- If curl fails: Check vLLM is running on RunPod (Step 6-7)
- If "port already in use": Kill existing tunnel: `pkill -f "8000:localhost:8000"`

### ⏱️ Minute 9: Start Backend & Frontend

**Terminal 1:**
```bash
cd /Users/omer/Desktop/ai-stanbul/backend
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

**Terminal 2:**
```bash
cd /Users/omer/Desktop/ai-stanbul/frontend
npm run dev
```

### ⏱️ Minute 10: Test!

Open: http://localhost:5173

Ask: "Merhaba! Istanbul hakkında bilgi ver."

Works? **SUCCESS!** 🎉

---

## 🌐 Make It Public (Bonus: 5 Minutes)

### Expose Backend
```bash
ngrok http 5000
```
Copy URL → Update `frontend/.env`:
```bash
echo "VITE_API_BASE_URL=https://YOUR-BACKEND-URL.ngrok-free.app" > frontend/.env
```

### Expose Frontend
```bash
npm run dev  # Restart frontend
ngrok http 5173  # In new terminal
```

**Share the frontend URL!** 🚀

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Permission denied" | Add SSH key (Step 1-2) |
| "Connection refused" | Restart pod (Step 3-4) |
| vLLM not responding | Check logs: `tail /root/vllm.log` |
| Tunnel fails | Run: `./setup_direct_tcp_tunnel.sh` |

---

## 📚 Detailed Guides

If you need more details:
- `RUNPOD_DEPLOYMENT_FINAL.md` - Complete step-by-step guide
- `ADD_SSH_KEY_TO_RUNPOD.md` - SSH setup help
- `RUNPOD_CONNECTION_TROUBLESHOOTING.md` - Troubleshooting help

---

## ✅ Success Checklist

- [ ] SSH key added to RunPod
- [ ] Pod restarted
- [ ] SSH connection works
- [ ] vLLM running on RunPod
- [ ] Tunnel created
- [ ] Backend started
- [ ] Frontend started
- [ ] Chatbot works locally
- [ ] (Optional) Exposed publicly

---

**Ready? Start with "Add SSH Key" and go! 🚀**
