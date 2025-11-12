# 🖥️ GPU Terminal Commands - RunPod Setup

## 1️⃣ Connect to Your RunPod GPU

Open RunPod terminal (Web Terminal or SSH)

---

## 2️⃣ Check if Server is Already Running

```bash
# Check if vllm server is running
ps aux | grep vllm

# If running, you'll see something like:
# python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B ...
```

---

## 3️⃣ If NOT Running, Start the Server

### **Option A: Start in Screen Session (Recommended)**

```bash
# Create a new screen session
screen -S vllm_server

# Start the vLLM server with 4-bit quantization
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B \
  --host 0.0.0.0 \
  --port 8888 \
  --quantization bitsandbytes \
  --load-format bitsandbytes

# Detach from screen: Press Ctrl+A, then D
# To reattach: screen -r vllm_server
```

### **Option B: Start Directly (No Screen)**

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B \
  --host 0.0.0.0 \
  --port 8888 \
  --quantization bitsandbytes \
  --load-format bitsandbytes
```

---

## 4️⃣ Wait for Server to Load

You should see output like:
```
INFO:     Loading model weights...
INFO:     Model loaded: meta-llama/Llama-3.1-8B
INFO:     Using 4-bit quantization
INFO:     GPU Memory: 3.2 GB / 16 GB
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8888
```

**Wait until you see:** `Uvicorn running on http://0.0.0.0:8888`

---

## 5️⃣ Test the Server (In GPU Terminal)

### **Test Health Endpoint:**
```bash
curl http://localhost:8888/health
```

Expected response:
```json
{"status":"healthy","model":"meta-llama/Llama-3.1-8B"}
```

### **Test Text Generation:**
```bash
curl http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "prompt": "Istanbul is a beautiful city known for",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

---

## 6️⃣ Verify Public Endpoint

The public endpoint should be:
```
https://4vq1b984pitw8s-8888.proxy.runpod.net
```

Test from your **local machine** (not GPU):
```bash
curl https://4vq1b984pitw8s-8888.proxy.runpod.net/health
```

---

## 7️⃣ Keep Server Running

If you used screen session:
```bash
# Detach from screen
Ctrl+A, then D

# Your server will keep running even if you close terminal

# To check later:
screen -ls          # List screens
screen -r vllm_server  # Reattach to server
```

---

## ✅ Checklist

- [ ] Connected to RunPod GPU terminal
- [ ] Started vLLM server (screen or direct)
- [ ] Waited for "Uvicorn running" message
- [ ] Tested local health endpoint (works)
- [ ] Tested public endpoint from your machine (works)
- [ ] Server running in screen (persistent)

---

## 🆘 Troubleshooting

### **Port Already in Use**
```bash
# Kill existing process
lsof -ti:8888 | xargs kill -9

# Start server again
```

### **Model Not Found**
```bash
# Download model first
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B
```

### **Out of Memory**
```bash
# Use smaller batch size
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B \
  --host 0.0.0.0 \
  --port 8888 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8
```

---

## 📝 Quick Commands Reference

```bash
# Start server in screen
screen -S vllm_server
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B --host 0.0.0.0 --port 8888 --quantization bitsandbytes --load-format bitsandbytes

# Detach: Ctrl+A, D

# Check if running
ps aux | grep vllm

# Test local
curl http://localhost:8888/health

# Test public (from local machine)
curl https://4vq1b984pitw8s-8888.proxy.runpod.net/health

# Reattach to screen
screen -r vllm_server

# Kill server
lsof -ti:8888 | xargs kill -9
```

---

**Once the server shows "Uvicorn running", test your backend!**

From your **local machine**:
```bash
cd /Users/omer/Desktop/ai-stanbul
python test_pure_llm_backend.py
```

All tests should pass! 🎉
