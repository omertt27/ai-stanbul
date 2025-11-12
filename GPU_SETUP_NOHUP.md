# 🚀 COPY-PASTE: GPU Setup (No Screen)

## Use this if `screen` command not found

---

### 1. Install Everything
```bash
pip install vllm huggingface_hub hf_transfer
```

---

### 2. Login
```bash
huggingface-cli login
```
Paste your token from: https://huggingface.co/settings/tokens

---

### 3. Download Model
```bash
huggingface-cli download meta-llama/Llama-3.1-8B
```
**Wait 5-10 minutes for download**

---

### 4. Start Server (Background)
```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B \
  --host 0.0.0.0 \
  --port 8888 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  > vllm_server.log 2>&1 &
```

---

### 5. Watch the Log
```bash
tail -f vllm_server.log
```

**Wait for:** `INFO:     Uvicorn running on http://0.0.0.0:8888`

**Then press:** `Ctrl+C` (stops viewing log, server keeps running)

---

### 6. Test Local
```bash
curl http://localhost:8888/health
```

**Should show:** `{"status":"healthy","model":"meta-llama/Llama-3.1-8B"}`

---

### 7. Test from Your Mac
```bash
curl https://4vq1b984pitw8s-8888.proxy.runpod.net/health
```

---

### 8. Run Full Test (on Your Mac)
```bash
cd /Users/omer/Desktop/ai-stanbul
python test_pure_llm_backend.py
```

**All 5 tests should pass!** ✅

---

## 🔍 Check Server Status

```bash
# Is it running?
ps aux | grep vllm

# View latest logs
tail -n 50 vllm_server.log

# Follow logs in real-time
tail -f vllm_server.log
```

---

## 🛑 Stop Server

```bash
pkill -f vllm
```

---

## ⚡ That's It!

Server runs in background and survives terminal close.

**Your Pure LLM backend is ready!** 🎉
