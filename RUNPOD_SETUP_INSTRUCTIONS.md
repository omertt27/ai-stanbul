# 🚀 RunPod GPU Setup Instructions

**Date:** November 23, 2025  
**Purpose:** Set up LLM service on RunPod GPU for AI Istanbul  
**Model:** Llama 3.1 8B Instruct (4-bit quantized)  
**Time Required:** 15-20 minutes

---

## 📋 Prerequisites

- ✅ RunPod GPU pod started (you've done this!)
- ✅ SSH access to your pod
- ✅ Backend deployed on Render (https://ai-stanbul.onrender.com)
- ✅ Frontend deployed on Vercel (https://aistanbul.net)

---

## 🔧 Step 1: Connect to RunPod (5 min)

### Option A: SSH Connection (Recommended)

1. **Get SSH Connection Details:**
   - Go to https://www.runpod.io/console/pods
   - Find your pod
   - Click **"Connect"** button
   - Copy the SSH command (looks like: `ssh root@<pod-id>.runpod.io -p <port> -i ~/.ssh/id_ed25519`)

2. **Connect via Terminal:**
   ```bash
   # Replace with your actual SSH command from RunPod
   ssh root@your-pod-id.runpod.io -p 12345 -i ~/.ssh/id_ed25519
   ```

### Option B: Web Terminal

1. Go to RunPod console
2. Click **"Connect"** → **"Start Web Terminal"**
3. Terminal opens in browser

---

## 📦 Step 2: Install Dependencies (5 min)

Once connected to your RunPod instance:

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install Python and essential tools
apt-get install -y python3-pip git wget curl

# Install PyTorch with CUDA support (if not already installed)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and required libraries
pip3 install transformers>=4.35.0
pip3 install accelerate>=0.24.0
pip3 install bitsandbytes>=0.41.0
pip3 install fastapi>=0.104.0
pip3 install uvicorn>=0.24.0
pip3 install pydantic>=2.0.0

# Verify GPU is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

**Expected output:**
```
CUDA available: True
GPU count: 1 (or more)
```

---

## 📥 Step 3: Download LLM Server Code (2 min)

### Option A: Copy from Local (Recommended)

If you have the code locally, upload it:

```bash
# From your local machine (in a NEW terminal window, NOT on RunPod)
cd /Users/omer/Desktop/ai-stanbul

# Upload the LLM server script to RunPod
scp -P <port> -i ~/.ssh/id_ed25519 llm_api_server_4bit.py root@your-pod-id.runpod.io:~/
```

### Option B: Create File Manually on RunPod

```bash
# On RunPod terminal
cat > llm_api_server_4bit.py << 'EOF'
# Paste the content of your llm_api_server_4bit.py file here
# (I can provide the full file if needed)
EOF
```

### Option C: Clone from GitHub (if you have a repo)

```bash
git clone https://github.com/your-username/ai-stanbul.git
cd ai-stanbul
```

---

## 🚀 Step 4: Start LLM Server (3 min)

```bash
# Navigate to the directory with the script
cd ~

# Start the LLM API server
python3 llm_api_server_4bit.py

# Alternative: Run in background with nohup
nohup python3 llm_api_server_4bit.py > llm_server.log 2>&1 &

# Check if it's running
ps aux | grep llm_api_server
```

**What happens:**
- Downloads Llama 3.1 8B model (4-bit quantized, ~5GB)
- Loads model into GPU memory
- Starts FastAPI server on port 8000

**Expected output:**
```
INFO: Loading model meta-llama/Llama-3.1-8B-Instruct with 4-bit quantization...
INFO: Model loaded successfully on cuda
INFO: Starting Uvicorn server on 0.0.0.0:8000
INFO: Application startup complete
```

---

## 🌐 Step 5: Expose Server via SSH Tunnel (2 min)

You need to create an SSH tunnel to access the LLM server from your backend.

### On Your Local Machine (NOT on RunPod):

```bash
# Open a NEW terminal window on your Mac
# Create SSH tunnel to forward RunPod port 8000 to localhost:8000
ssh -L 8000:localhost:8000 -p <runpod-port> -i ~/.ssh/id_ed25519 root@your-pod-id.runpod.io -N
```

**What this does:**
- Maps RunPod's port 8000 → Your local port 8000
- Keeps connection alive (don't close this terminal!)

### Alternative: Use RunPod Public URL

RunPod can provide a public HTTP endpoint:

1. In RunPod console, click your pod
2. Look for **"TCP Port Mappings"**
3. Find port 8000 → Note the public URL
4. Use this URL in your backend config

---

## ✅ Step 6: Test LLM Server (3 min)

### Test 1: Health Check

```bash
# On your local machine (in another terminal)
curl http://localhost:8000/
```

**Expected:**
```json
{
  "name": "AI Istanbul LLM API (4-bit)",
  "version": "4.0",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "status": "ready"
}
```

### Test 2: Generate Response

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the best restaurants in Sultanahmet?",
    "max_tokens": 200,
    "temperature": 0.7
  }' | jq .
```

**Expected:**
```json
{
  "generated_text": "Here are some excellent restaurants in Sultanahmet:\n\n1. Tarihi Sultanahmet Köftecisi - Famous meatballs...",
  "generation_time": 2.34,
  "tokens_generated": 156
}
```

---

## 🔗 Step 7: Connect Backend to LLM (5 min)

Now update your Render backend to use the LLM service.

### Option A: Using SSH Tunnel (Local Development)

If you're running backend locally:

```bash
# In backend/.env or environment variables
LLM_API_URL=http://localhost:8000
LLM_API_KEY=optional_key_here
```

### Option B: Using RunPod Public URL (Production)

1. Get RunPod public URL for port 8000
2. Update Render environment variables:

```bash
# Go to Render dashboard → Your service → Environment
# Add these variables:

LLM_API_URL=http://your-runpod-url:8000
# or
OPENAI_API_BASE_URL=http://your-runpod-url:8000
OPENAI_API_KEY=not-needed-for-local-model
```

3. Save and redeploy

---

## 🧪 Step 8: Test End-to-End (5 min)

### Test Backend → LLM Connection

```bash
# Test your backend's chat endpoint
curl -X POST https://ai-stanbul.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What should I visit in Istanbul?",
    "language": "en"
  }' | jq .
```

### Test Frontend → Backend → LLM

1. Go to https://aistanbul.net
2. Open chat
3. Type: "Tell me about Hagia Sophia"
4. Should get AI-generated response (not template!)

---

## 📊 Monitoring & Logs

### View LLM Server Logs

```bash
# If running in background with nohup
tail -f ~/llm_server.log

# If running in foreground, logs appear in terminal
```

### Check GPU Usage

```bash
# On RunPod
nvidia-smi

# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```

**Expected:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  RTX 4090        Off      | 00000000:01:00.0 Off |                  N/A |
| 30%   45C    P0    78W / 450W |   6215MiB / 24564MiB |     12%      Default |
+-------------------------------+----------------------+----------------------+
```

---

## 🔧 Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Use 4-bit quantization (should already be enabled)
# Or reduce batch size in the code
# Or use a smaller model
```

### Issue: "Model download failed"

**Solution:**
```bash
# Check internet connection
ping huggingface.co

# Manually download model
python3 -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')"
```

### Issue: "Port 8000 already in use"

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
python3 llm_api_server_4bit.py --port 8001
```

### Issue: "SSH tunnel keeps disconnecting"

**Solution:**
```bash
# Add keep-alive to SSH config
# Edit ~/.ssh/config on your Mac:
cat >> ~/.ssh/config << EOF
Host runpod
    HostName your-pod-id.runpod.io
    Port <port>
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF

# Then connect with:
ssh -L 8000:localhost:8000 runpod -N
```

---

## 🎯 Alternative: Use Groq API (Simpler)

If RunPod setup is complex, you can use Groq's free API instead:

```bash
# Get free API key from https://console.groq.com

# Add to Render environment:
GROQ_API_KEY=your_groq_api_key_here

# Backend will automatically use Groq instead of local LLM
```

**Pros:**
- No GPU management
- No SSH tunnels
- Fast inference
- Free tier available

**Cons:**
- Rate limits
- External dependency
- Less customization

---

## 📋 Quick Command Reference

```bash
# Connect to RunPod
ssh root@your-pod-id.runpod.io -p <port> -i ~/.ssh/id_ed25519

# Start LLM server
python3 llm_api_server_4bit.py

# Start LLM in background
nohup python3 llm_api_server_4bit.py > llm_server.log 2>&1 &

# Create SSH tunnel (local machine)
ssh -L 8000:localhost:8000 -p <port> -i ~/.ssh/id_ed25519 root@your-pod-id.runpod.io -N

# Test LLM
curl http://localhost:8000/

# Test chat
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_tokens":50}'

# View logs
tail -f ~/llm_server.log

# Check GPU
nvidia-smi

# Stop LLM server
pkill -f llm_api_server
```

---

## 🎊 Success Checklist

After completing these steps, verify:

- [ ] RunPod GPU pod is running
- [ ] LLM server is running on port 8000
- [ ] GPU memory shows model loaded (~6GB)
- [ ] Health endpoint returns "ready"
- [ ] Test generation works
- [ ] SSH tunnel is active (if using)
- [ ] Backend can connect to LLM
- [ ] Frontend chat shows AI responses

---

## 📞 Next Steps

Once LLM is running:

1. ✅ Complete the final 2% of Phase 4:
   - Fix Vercel API paths
   - Update Render CORS
   - Test integration

2. 🎉 Celebrate 100% deployment!

3. 🚀 Optional enhancements:
   - Set up auto-restart for LLM server
   - Configure monitoring
   - Optimize inference speed
   - Add caching layer

---

## 💡 Pro Tips

1. **Keep SSH tunnel alive:** Use `autossh` or `tmux`
2. **Monitor costs:** RunPod charges by the hour
3. **Save checkpoints:** Stop pod when not in use
4. **Use templates:** RunPod has pre-configured ML templates
5. **Consider Groq:** For production, managed API is easier

---

**Need help?** Check:
- RunPod docs: https://docs.runpod.io
- Llama 3 docs: https://huggingface.co/meta-llama
- Your project's `llm_api_server_4bit.py`

**Ready to start?** Follow Step 1! 🚀
