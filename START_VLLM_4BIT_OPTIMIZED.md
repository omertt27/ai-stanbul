# 🚀 Start vLLM 4-bit AWQ Model - Optimized Settings

## ✅ Step-by-Step Commands

### **Step 1: Verify Model is Downloaded**

```bash
ls -lh /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/
```

**Expected:** You should see files like `config.json`, `tokenizer.json`, `*.safetensors`

**If model is missing, download it:**
```bash
export HF_TOKEN=AISTANBUL
export HF_HOME=/workspace/.cache

huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --local-dir /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --cache-dir /workspace/.cache
```

---

### **Step 2: Check GPU Memory**

```bash
nvidia-smi
```

**Note:** How much GPU memory is available? This helps choose the right settings.

---

### **Step 3: Start vLLM (Recommended Settings)**

```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.75 \
  --max-model-len 2048 \
  --port 8000 \
  --host 0.0.0.0 \
  --enforce-eager \
  > /workspace/vllm.log 2>&1 &

echo "✅ vLLM starting..."
echo "📝 Logs: /workspace/vllm.log"
```

**Settings Explained:**
- `--quantization awq` - Use 4-bit AWQ quantization
- `--gpu-memory-utilization 0.75` - Use 75% of GPU memory (conservative)
- `--max-model-len 2048` - Context window of 2048 tokens (good balance)
- `--port 8000` - Use port 8000
- `--enforce-eager` - Disable CUDA graphs (more stable, slightly slower)

---

### **Step 4: Wait and Check Logs**

```bash
# Wait 30 seconds for initialization
sleep 30

# Watch logs (press Ctrl+C to stop)
tail -f /workspace/vllm.log
```

**Look for:**
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### **Step 5: Test Locally**

```bash
# Test health endpoint
curl http://localhost:8000/health

# Should return: {"status": "ok"} or similar

# Test v1/models
curl http://localhost:8000/v1/models

# Test chat completion
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

**Expected:** JSON response with AI-generated Turkish greeting

---

## 🌐 Your Public Endpoint

Once vLLM is running successfully:

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
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

---

## ⚙️ Alternative Settings (If Above Doesn't Work)

### **Ultra-Conservative (For GPUs with < 16GB)**

```bash
pkill -f vllm

nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.60 \
  --max-model-len 1024 \
  --port 8000 \
  --host 0.0.0.0 \
  --enforce-eager \
  --max-num-seqs 8 \
  > /workspace/vllm.log 2>&1 &
```

### **Aggressive (For GPUs with >= 24GB)**

```bash
pkill -f vllm

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

## 🔍 Monitoring

### **Check if vLLM is Running**

```bash
ps aux | grep vllm | grep -v grep
```

### **View Live Logs**

```bash
tail -f /workspace/vllm.log
```

### **Check GPU Usage**

```bash
nvidia-smi
```

### **Check Last 50 Lines of Logs**

```bash
tail -50 /workspace/vllm.log
```

---

## 🎯 Success Checklist

- [ ] Model files exist in `/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/`
- [ ] vLLM process is running (`ps aux | grep vllm`)
- [ ] Logs show "Application startup complete"
- [ ] Local health check returns success (`curl http://localhost:8000/health`)
- [ ] Local chat completion works
- [ ] Public endpoint accessible from Mac

---

## 🚀 Next Step: Update Render

Once vLLM is confirmed working:

1. **Go to Render Dashboard:** https://dashboard.render.com
2. **Select:** ai-stanbul backend service
3. **Go to:** Environment tab
4. **Update `LLM_API_URL` to:**
   ```
   https://i6c58scsmccj2s-8000.proxy.runpod.net/v1
   ```
5. **Save** (triggers auto-redeploy)
6. **Test chat app** - should get real AI responses! 🎉

---

## 📊 Quick Diagnostic One-Liner

Run this to check everything at once:

```bash
echo "=== CHECKING vLLM STATUS ===" && \
echo "Process:" && ps aux | grep vllm | grep -v grep || echo "❌ Not running" && \
echo -e "\nGPU:" && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader && \
echo -e "\nHealth:" && curl -s http://localhost:8000/health 2>/dev/null || echo "❌ Not responding" && \
echo -e "\nLast 10 log lines:" && tail -10 /workspace/vllm.log
```

---

**Ready? Run Step 3 to start vLLM!** 🚀
