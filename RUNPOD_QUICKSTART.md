# 🚀 Connect AI Istanbul to Your RunPod RTX A5000

## Quick Start (3 Steps)

### 1️⃣ Deploy on RunPod (5 minutes)

```bash
# SSH into your RunPod
ssh fgkqzve33ssbea-64411271@ssh.runpod.io -i ~/.ssh/id_ed25519

# Download and run the deploy script
curl -sSL https://raw.githubusercontent.com/YOUR-REPO/ai-stanbul/main/deploy_vllm_runpod.sh | bash

# OR manually:
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000 --dtype float16
```

### 2️⃣ Get Your Endpoint URL

Go to https://www.runpod.io/console/pods and copy your pod's proxy URL:
```
https://fgkqzve33ssbea-8000.proxy.runpod.net
```

### 3️⃣ Configure AI Istanbul (1 minute)

```bash
# On your local machine
cd /Users/omer/Desktop/ai-stanbul
./setup_runpod_connection.sh
# Enter your RunPod URL when prompted
```

**OR manually edit `.env`:**
```bash
echo 'LLM_API_URL=https://fgkqzve33ssbea-8000.proxy.runpod.net/v1' >> backend/.env
```

---

## 🧪 Test Everything

```bash
# Test 1: Test RunPod connection
cd backend
python test_runpod_connection.py

# Test 2: Start backend
python api_server.py

# Test 3: Test health endpoint
curl http://localhost:8001/api/v1/llm/health

# Test 4: Test generation
curl -X POST http://localhost:8001/api/v1/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Istanbul?", "max_tokens": 100}'
```

---

## 📊 Monitor Performance

### Admin Dashboard
- URL: http://localhost:3000/admin
- Username: `KAM`
- Password: (your admin password)

View real-time metrics:
- ✅ Total LLM queries
- ✅ Cache hit rate
- ✅ Average response time
- ✅ Error rates

### RunPod GPU Monitoring

```bash
# SSH into RunPod
ssh fgkqzve33ssbea-64411271@ssh.runpod.io -i ~/.ssh/id_ed25519

# Watch GPU usage
watch -n 1 nvidia-smi
```

---

## 🔧 Configuration Options

Edit `backend/.env`:

```bash
# Your RunPod endpoint (REQUIRED)
LLM_API_URL=https://YOUR-POD-ID-8000.proxy.runpod.net/v1

# Optional: API key for authentication
RUNPOD_API_KEY=your_api_key_here

# Timeout for requests (seconds)
LLM_TIMEOUT=60

# Maximum tokens per request
LLM_MAX_TOKENS=250

# Temperature (0.0 = deterministic, 1.0 = creative)
LLM_TEMPERATURE=0.7
```

---

## 🚀 Advanced: Running vLLM in Background

### Option 1: Using tmux (Simple)

```bash
# On RunPod
tmux new -s vllm
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000
# Press Ctrl+B, then D to detach
# Reattach later: tmux attach -t vllm
```

### Option 2: Using Docker (Persistent)

```bash
docker run -d \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name vllm-server \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype float16 \
  --gpu-memory-utilization 0.85
```

---

## 📈 Expected Performance (RTX A5000)

| Metric | Value |
|--------|-------|
| Model | Llama 3.1 8B |
| Tokens/second | 40-60 |
| Latency (200 tokens) | 3-5 seconds |
| Memory Usage | 8-10 GB VRAM |
| Max Context Length | 4096 tokens |
| Concurrent Requests | 10-30 |

---

## 🐛 Troubleshooting

### "Connection refused"

✅ **Solution:**
```bash
# Check if vLLM is running on RunPod
ssh fgkqzve33ssbea-64411271@ssh.runpod.io -i ~/.ssh/id_ed25519
ps aux | grep vllm
curl http://localhost:8000/v1/models
```

### "LLM_API_URL not configured"

✅ **Solution:**
```bash
# Add to backend/.env
echo 'LLM_API_URL=https://YOUR-POD-URL/v1' >> backend/.env
```

### "Timeout error"

✅ **Solution:**
```bash
# Increase timeout in backend/.env
LLM_TIMEOUT=120
```

### "Out of memory on RunPod"

✅ **Solution:**
```bash
# Reduce GPU memory usage
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.75  # Reduced from 0.85
```

---

## 📚 Documentation

- **Full Setup Guide:** [RUNPOD_SETUP_GUIDE.md](./RUNPOD_SETUP_GUIDE.md)
- **RunPod Docs:** https://docs.runpod.io/
- **vLLM Docs:** https://docs.vllm.ai/
- **Admin Dashboard:** http://localhost:3000/admin

---

## 🎯 What You Get

✅ **High-Performance LLM:** 40-60 tokens/s on RTX A5000  
✅ **Cost-Effective:** ~$0.10-0.34/hour on RunPod  
✅ **OpenAI-Compatible API:** Drop-in replacement  
✅ **Real-Time Analytics:** Monitor everything in admin dashboard  
✅ **Smart Caching:** Reduce API calls by 50-80%  
✅ **Multi-Language:** English & Turkish support  

---

**Your AI Istanbul system is ready to leverage your RTX A5000! 🚀🇹🇷**
