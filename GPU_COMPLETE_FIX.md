# 🚀 GPU LLM Server - Complete Installation & Fix

## The Problem

You're getting: **"Please install bitsandbytes>=0.46.1 via `pip install bitsandbytes>=0.46.1`"**

This means the `bitsandbytes` package (required for 4-bit quantization) is missing.

---

## ✅ Complete Fix (One Command)

```bash
pip install bitsandbytes>=0.46.1 && pkill -f vllm; sleep 2 && nohup python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --quantization bitsandbytes --load-format bitsandbytes --host 0.0.0.0 --port 8888 --max-model-len 8192 > vllm.log 2>&1 & sleep 5 && tail -f vllm.log
```

**This will:**
1. ✅ Install `bitsandbytes` (required for 4-bit quantization)
2. ✅ Kill any existing vLLM server on port 8888
3. ✅ Wait 2 seconds
4. ✅ Start new vLLM server with 4-bit quantization
5. ✅ Wait 5 seconds
6. ✅ Show logs (you'll see model loading progress)

---

## 📋 Step-by-Step (If You Prefer)

### Step 1: Install bitsandbytes

```bash
pip install bitsandbytes>=0.46.1
```

### Step 2: Verify Installation

```bash
python -c "import bitsandbytes; print('✅ bitsandbytes installed:', bitsandbytes.__version__)"
```

You should see: `✅ bitsandbytes installed: 0.46.1` (or higher)

### Step 3: Kill Old Server

```bash
pkill -f vllm
```

Wait 2 seconds.

### Step 4: Start New Server

```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --host 0.0.0.0 \
  --port 8888 \
  --max-model-len 8192 \
  > vllm.log 2>&1 &
```

### Step 5: Monitor Logs

```bash
tail -f vllm.log
```

---

## ✅ Success Indicators

**Look for these in the logs:**

1. ✅ `Initializing an LLM engine with config...`
2. ✅ `Loading model weights from meta-llama/Llama-3.1-8B-Instruct...`
3. ✅ `Using bitsandbytes quantization...`
4. ✅ `Uvicorn running on http://0.0.0.0:8888`

---

## 🧪 Test the Server

### Test 1: Health Check (Local)

```bash
curl http://localhost:8888/health
```

Expected: `{"status": "ok"}`

### Test 2: Health Check (Public RunPod Proxy)

```bash
curl https://YOUR-POD-ID-8888.proxy.runpod.net/health
```

Expected: `{"status": "ok"}`

### Test 3: Generate Text

```bash
curl -X POST http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Istanbul is a city in",
    "max_tokens": 50
  }'
```

---

## 🔧 Troubleshooting

### "bitsandbytes import failed"

Upgrade pip first, then reinstall:

```bash
pip install --upgrade pip
pip install --upgrade bitsandbytes>=0.46.1
```

### "Port 8888 is already in use"

Kill all vLLM processes:

```bash
pkill -9 -f vllm
```

Wait 5 seconds, then start server again.

### "Model not found" or "Access denied"

Make sure you're logged into Hugging Face:

```bash
huggingface-cli login
```

Enter your token (get it from https://huggingface.co/settings/tokens)

### "CUDA out of memory"

Reduce max model length:

```bash
--max-model-len 4096
```

Or use a smaller model:

```bash
--model meta-llama/Llama-3.1-8B-Instruct  # (This is already the 8B version)
```

---

## 🎯 Quick Reference

### Check if server is running

```bash
ps aux | grep vllm
```

### View logs

```bash
tail -f vllm.log
```

### Stop server

```bash
pkill -f vllm
```

### Restart server

```bash
pkill -f vllm; sleep 2 && nohup python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --quantization bitsandbytes --load-format bitsandbytes --host 0.0.0.0 --port 8888 --max-model-len 8192 > vllm.log 2>&1 & sleep 5 && tail -f vllm.log
```

---

## 🚀 Next Steps

Once the server is running successfully:

1. ✅ Get your public RunPod proxy URL (format: `https://YOUR-POD-ID-8888.proxy.runpod.net`)
2. ✅ Update `.env` with the correct endpoint:
   ```
   RUNPOD_LLM_ENDPOINT=https://YOUR-POD-ID-8888.proxy.runpod.net
   ```
3. ✅ Test the backend integration:
   ```bash
   python test_pure_llm_backend.py
   ```
4. ✅ Start the backend:
   ```bash
   cd backend && uvicorn main_pure_llm:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## 📚 Related Guides

- **Full Setup**: `GPU_LLM_SETUP_GUIDE.md`
- **Quick Start**: `GPU_SETUP_SIMPLE.md`
- **Persistent Server**: `GPU_SETUP_NOHUP.md`
- **Backend Testing**: `test_pure_llm_backend.py`

---

**Now go ahead and run the complete fix command above! 🚀**
