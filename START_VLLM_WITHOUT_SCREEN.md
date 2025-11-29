# 🚀 Start vLLM Without Screen (nohup method)

Since `screen` is not installed on your RunPod pod, we'll use `nohup` to run vLLM in the background.

---

## ✅ Commands to Run (Copy-Paste These)

### **Step 1: Check if Model is Downloaded**

```bash
ls -lh /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/
```

**If the directory doesn't exist or is empty**, download the model first:

```bash
export HF_TOKEN=AISTANBUL
export HF_HOME=/workspace/.cache

huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --local-dir /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --cache-dir /workspace/.cache
```

---

### **Step 2: Start vLLM with nohup**

```bash
# Start vLLM in background on port 8000
nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8000 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &

echo "✅ vLLM starting in background..."
echo "📝 Logs: /workspace/vllm.log"
echo "🔍 Check status: tail -f /workspace/vllm.log"
```

**What this does:**
- Starts vLLM on **port 8000**
- Runs in background (you can close terminal)
- Logs output to `/workspace/vllm.log`
- The `&` at the end makes it run in background

---

### **Step 3: Wait and Check Logs**

```bash
# Watch the logs (wait for "Application startup complete")
tail -f /workspace/vllm.log
```

**What to look for:**
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Press `Ctrl+C` to stop watching logs** (vLLM keeps running in background)

---

### **Step 4: Verify vLLM is Running**

```bash
# Check if process is running
ps aux | grep vllm | grep -v grep

# Test health endpoint locally
curl http://localhost:8000/health

# Should return something like: {"status": "ok"}
```

---

### **Step 5: Test Chat Completion Locally**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "messages": [
      {"role": "user", "content": "Say hello in Turkish"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**If you see JSON with AI-generated text, it's working! ✅**

---

## 🌐 Your Public Endpoint

Once vLLM is running, your public endpoint will be:

```
https://i6c58scsmccj2s-8000.proxy.runpod.net/v1
```

---

## 🧪 Test Public Endpoint (From Your Mac)

Open a **new terminal on your Mac** and run:

```bash
# Test health
curl https://i6c58scsmccj2s-8000.proxy.runpod.net/health

# Test chat
curl -X POST https://i6c58scsmccj2s-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "messages": [
      {"role": "user", "content": "What are the top 3 places to visit in Istanbul?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

---

## 🔧 Managing vLLM Process

### **Check if vLLM is Running**

```bash
ps aux | grep vllm | grep -v grep
```

### **View Live Logs**

```bash
tail -f /workspace/vllm.log
```

### **Stop vLLM**

```bash
pkill -f vllm
```

### **Restart vLLM**

```bash
# Stop it first
pkill -f vllm

# Start again
nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8000 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &
```

---

## 📊 Monitor GPU Usage

```bash
# Check GPU memory usage
nvidia-smi

# Watch in real-time (press Ctrl+C to stop)
watch -n 2 nvidia-smi
```

---

## ⚠️ Troubleshooting

### **Issue: "Port already in use"**

```bash
# Find what's using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>
```

### **Issue: "CUDA out of memory"**

Edit the startup command and reduce memory:
```bash
--gpu-memory-utilization 0.7  # Instead of 0.85
--max-model-len 2048  # Instead of 4096
```

### **Issue: "Model not found"**

Download the model:
```bash
export HF_TOKEN=AISTANBUL
export HF_HOME=/workspace/.cache

huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --local-dir /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
```

### **Issue: Can't see logs**

```bash
tail -100 /workspace/vllm.log
```

---

## ✅ Quick Start Summary

**Just copy-paste these 3 commands:**

```bash
# 1. Start vLLM
nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8000 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &

# 2. Wait 30 seconds, then check logs
tail -f /workspace/vllm.log

# 3. Test it (in new terminal or after Ctrl+C from logs)
curl http://localhost:8000/health
```

---

## 🎯 Next Step: Update Render

Once vLLM is running and responding to `curl http://localhost:8000/health`:

1. **Update Render Environment Variable:**
   - Go to: https://dashboard.render.com
   - Service: **ai-stanbul**
   - Environment tab
   - Update `LLM_API_URL` to:
     ```
     https://i6c58scsmccj2s-8000.proxy.runpod.net/v1
     ```
   - Save (auto-redeploys)

2. **Test your chat app!** 🎉

---

**Let me know once you run the nohup command and I'll help you verify it's working!**
