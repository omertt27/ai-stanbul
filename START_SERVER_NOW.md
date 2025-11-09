# 🚀 START SERVER - Quick Commands

## One Command to Rule Them All

### SSH and Start Server:

```bash
gcloud compute ssh instance-20251109-085407 --zone=europe-west1-b -- 'cd ~/ai-istanbul && chmod +x setup_and_start_llama.sh && ./setup_and_start_llama.sh'
```

---

## OR: Two Commands (Recommended for Monitoring)

### Terminal 1: Start Server

```bash
gcloud compute ssh instance-20251109-085407 --zone=europe-west1-b
cd ~/ai-istanbul
./setup_and_start_llama.sh
```

### Terminal 2: Monitor Logs

```bash
gcloud compute ssh instance-20251109-085407 --zone=europe-west1-b
tail -f /tmp/llm_api_server.log
```

---

## What the Script Does:

1. ✅ Sets up HuggingFace authentication
2. ✅ Installs required dependencies
3. ✅ Verifies authentication
4. ✅ Stops any existing server
5. ✅ Starts Llama 3.1 8B server

---

## Expected Output:

```
============================================================
🚀 AI Istanbul - Llama 3.1 8B Server Setup
============================================================

Step 1/3: Setting up HuggingFace Authentication...
✅ Authentication configured

Step 2/3: Installing HuggingFace Hub...
✅ Dependencies installed

Step 3/3: Verifying authentication...
✅ Authenticated as: AISTANBUL

Stopping any existing server...
✅ Ready to start

============================================================
🚀 Starting Llama 3.1 8B Server
============================================================

📊 Expected timeline:
   - Tokenizer loading: 10-20 seconds
   - Model download: 3-5 minutes (first time, ~15GB)
   - Model loading: 2-3 minutes
   - Total: 5-8 minutes (first time)

Starting server now...
```

Then you'll see the server logs:

```
2025-11-09 13:XX:XX - INFO - 🚀 Loading Open-Access LLM Model (CPU Mode)
2025-11-09 13:XX:XX - INFO - 💻 System Resources:
2025-11-09 13:XX:XX - INFO -    - CPUs: 8
2025-11-09 13:XX:XX - INFO -    - RAM: 31.4 GB
2025-11-09 13:XX:XX - INFO - 📥 Attempting to load: meta-llama/Meta-Llama-3.1-8B
2025-11-09 13:XX:XX - INFO -    Loading tokenizer...
2025-11-09 13:XX:XX - INFO -    Loading model...
2025-11-09 13:XX:XX - INFO -    ⚙️  Configuration: CPU-only, float32
[Downloading... this takes 3-5 minutes first time]
2025-11-09 13:XX:XX - INFO - ✅ Successfully loaded: meta-llama/Meta-Llama-3.1-8B
2025-11-09 13:XX:XX - INFO - 📊 Model: meta-llama/Meta-Llama-3.1-8B
2025-11-09 13:XX:XX - INFO - 📊 Parameters: 8.03B
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## Test from Local Machine (After Server Starts):

### Health Check:
```bash
curl http://35.210.251.24:8000/health
```

### Test Generation:
```bash
curl -X POST http://35.210.251.24:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the top 3 must-visit places in Istanbul?",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

### Run Full Test Suite:
```bash
cd /Users/omer/Desktop/ai-stanbul
python test_llm_integration.py --suite all
```

---

## 🎯 Ready to Start!

Just copy and paste the command from the top:

```bash
gcloud compute ssh instance-20251109-085407 --zone=europe-west1-b
cd ~/ai-istanbul
./setup_and_start_llama.sh
```

**That's it! The script handles everything! 🚀**
