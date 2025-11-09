# 🚀 Option A: Start API Server (Foreground Mode)

**Created:** November 9, 2025  
**Purpose:** Start the Llama 3.1 8B API server and verify all prompts/handlers use the new LLM

---

## ✅ STEP 1: START THE API SERVER

Open your terminal and run these commands:

```bash
# SSH into your Google Cloud VM
gcloud compute ssh instance-20251109-085407 --zone=europe-west1-b

# Navigate to project directory
cd ~/ai-stanbul

# Activate virtual environment
source venv/bin/activate

# Set environment variable
export ENVIRONMENT=production

# Start the API server (FOREGROUND MODE)
python3 llm_api_server.py
```

**Expected Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
🚀 Loading Llama 3.1 8B model...
Loading checkpoint shards: 100%|████████████| 4/4 [00:45<00:00, 11.32s/it]
✅ Model loaded successfully!
📊 Model size: 8.03B parameters
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**⚠️ Important:**
- Model loading takes 2-3 minutes
- Server will use ~20GB RAM
- CPU usage will be 100% during inference (normal)
- Keep this terminal window open (foreground mode)

---

## ✅ STEP 2: TEST THE API ENDPOINTS

Open a **NEW terminal window** on your Mac and run these tests:

### Test 1: Health Check
```bash
curl http://35.210.251.24:8000/health
```

**Expected:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### Test 2: Simple Generation
```bash
curl -X POST http://35.210.251.24:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Istanbul is",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Expected:**
```json
{
  "response": "a beautiful city that bridges Europe and Asia...",
  "tokens_generated": 45
}
```

### Test 3: Chat Interface
```bash
curl -X POST http://35.210.251.24:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the top 3 places to visit in Istanbul?",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

**Expected:**
```json
{
  "response": "Here are the top 3 must-visit places in Istanbul:\n\n1. Hagia Sophia - A stunning Byzantine masterpiece...",
  "tokens_generated": 150
}
```

---

## ✅ STEP 3: VERIFY ALL PROMPTS USE NEW LLM

All endpoints should now use the Llama 3.1 8B model. Here's what's been configured:

### ✅ LLM API Server (`llm_api_server.py`)
- **Model:** Llama 3.1 8B
- **Endpoints:** `/health`, `/generate`, `/chat`
- **Status:** ✅ Using new LLM

### ✅ Integration Wrapper (`integration_wrapper.py`)
- **Client:** GoogleCloudLLMClient
- **Calls:** Points to VM API (http://35.210.251.24:8000)
- **Status:** ✅ Routes to new LLM

### ✅ Main System (to be updated)
Now we need to update your main AI Istanbul system to use the new LLM for ALL prompts.

---

## 🔧 STEP 4: UPDATE ALL HANDLERS TO USE NEW LLM

I'll now update your system files to ensure ALL prompts and handlers use the new Llama 3.1 8B LLM:

### Files to Update:
1. ✅ `ml_systems/llm_service_wrapper.py` - Main LLM wrapper
2. ✅ `advanced_istanbul_ai.py` - Advanced AI system
3. ✅ `istanbul_ai/intent_classifier.py` - Intent classification
4. ✅ `istanbul_ai/response_generator.py` - Response generation
5. ✅ `ml_systems/multi_intent_handler.py` - Multi-intent handling

---

## 📋 MONITORING CHECKLIST

While the server is running, monitor:

- [ ] **Memory Usage:** Should stabilize at ~20GB
- [ ] **Response Time:** 3-8 seconds per request (acceptable)
- [ ] **CPU Usage:** 100% during inference (normal)
- [ ] **Logs:** Check for errors in terminal output
- [ ] **Network:** Ensure port 8000 is accessible

---

## 🛑 HOW TO STOP THE SERVER

When you want to stop the server:

1. Go to the SSH terminal window
2. Press `CTRL+C`
3. Wait for graceful shutdown

**Output:**
```
^C
INFO:     Shutting down
INFO:     Finished server process [12345]
```

---

## 🔄 NEXT STEPS AFTER SERVER IS RUNNING

1. ✅ Verify health endpoint works
2. ✅ Test generation endpoint
3. ✅ Test chat endpoint
4. ✅ Update all system handlers (I'll do this next)
5. ✅ Deploy to Render
6. ✅ Connect Vercel frontend
7. ✅ Test end-to-end flow

---

**Status:** 🚀 Ready to start!  
**Action:** Run the commands in STEP 1 now!  
**Next:** I'll update all handlers to use the new LLM while you start the server.
