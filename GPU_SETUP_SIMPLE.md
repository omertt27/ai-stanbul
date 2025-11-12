# 🚀 GPU Setup - Copy & Paste Commands

## Just copy these commands into your RunPod GPU terminal:

---

### Step 1: Install Dependencies
```bash
pip install vllm huggingface_hub hf_transfer
```

---

### Step 2: Login to Hugging Face
```bash
huggingface-cli login
```

**Paste your token when prompted**
- Get token from: https://huggingface.co/settings/tokens
- Make sure you've accepted the Llama license at: https://huggingface.co/meta-llama/Llama-3.1-8B

---

### Step 3: Download Model
```bash
huggingface-cli download meta-llama/Llama-3.1-8B
```

**Wait 5-10 minutes** for download to complete.

---

### Step 4: Start Server (Choose One Method)

**Method A: With screen (recommended if available)**
```bash
# Install screen first
apt-get update && apt-get install -y screen

# Start in screen
screen -S vllm_server
```

Then run:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B \
  --host 0.0.0.0 \
  --port 8888 \
  --quantization bitsandbytes \
  --load-format bitsandbytes
```

**Wait for:** `INFO:     Uvicorn running on http://0.0.0.0:8888`

---

### Step 5A: Detach from Screen (if using screen)
Press: **Ctrl+A**, then press **D**

---

**OR**

**Method B: With nohup (if screen not working)**
```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B \
  --host 0.0.0.0 \
  --port 8888 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  > vllm_server.log 2>&1 &
```

Check the log:
```bash
tail -f vllm_server.log
```

**Wait for:** `INFO:     Uvicorn running on http://0.0.0.0:8888`

Press **Ctrl+C** to stop viewing log (server keeps running)

---

### Step 6: Test It Works
```bash
curl http://localhost:8888/health
```

**Expected:** `{"status":"healthy","model":"meta-llama/Llama-3.1-8B"}`

---

### Step 7: Test from Your Local Machine

Open terminal on your Mac and run:
```bash
curl https://4vq1b984pitw8s-8888.proxy.runpod.net/health
```

**Expected:** `{"status":"healthy","model":"meta-llama/Llama-3.1-8B"}`

---

### Step 8: Test Full Backend

On your Mac:
```bash
cd /Users/omer/Desktop/ai-stanbul
python test_pure_llm_backend.py
```

**All 5 tests should pass!** ✅

---

## ⚡ If You See Deprecation Warning

If you see: `⚠️ Warning: 'huggingface-cli login' is deprecated`

Use the new commands instead:
```bash
# Login (NEW)
hf auth login

# Download (NEW)
hf download meta-llama/Llama-3.1-8B
```

Both old and new commands work fine!

---

## 🔄 Useful Commands

**If using screen:**
```bash
# Reattach to server
screen -r vllm_server

# List screens
screen -ls

# Detach: Ctrl+A, then D
```

**If using nohup:**
```bash
# View log
tail -f vllm_server.log

# Check if running
ps aux | grep vllm

# Stop server
pkill -f vllm
```

**General:**
```bash
# Check if running
ps aux | grep vllm

# Kill server
pkill -f vllm
```

---

## ⏱️ Estimated Time

- Install: 1-2 min
- Login: 30 sec
- Download: 5-10 min
- Start server: 2-3 min
- **Total: ~10-15 minutes**

---

## ✅ You're Done When...

You see this in your GPU terminal:
```
INFO:     Uvicorn running on http://0.0.0.0:8888
```

And this from your Mac:
```bash
$ curl https://4vq1b984pitw8s-8888.proxy.runpod.net/health
{"status":"healthy","model":"meta-llama/Llama-3.1-8B"}
```

🎉 **Your Pure LLM backend is now fully operational!**
