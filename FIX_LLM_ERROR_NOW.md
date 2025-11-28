# 🚨 FIX LLM ERROR - Step by Step

## What's Wrong?
Your Istanbul AI chatbot gives errors because **port 8000 is NOT exposed**.

Looking at your RunPod pod, you only have:
- ✅ Port 8888 (Jupyter Lab) 
- ✅ Port 22 (SSH)
- ❌ **Port 8000 MISSING!**

## 🎯 2-Step Fix

---

## STEP 1: Start vLLM (5 minutes)

### Option A: Via Web Terminal (Easiest)

1. **In your RunPod pod page**, click **"Open Web Terminal"** button

2. **Copy-paste this command** (all at once):
   ```bash
   pkill -9 -f vllm && python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000 --host 0.0.0.0 --dtype auto --max-model-len 4096 --gpu-memory-utilization 0.9 > /root/vllm.log 2>&1 &
   ```

3. **Wait 30 seconds**, then test:
   ```bash
   curl http://localhost:8000/health
   ```
   
   ✅ Should return: `{"status":"ok"}`

### Option B: Via Jupyter Lab

1. **Open**: https://pvj233wwhiu6j3-8888.proxy.runpod.net

2. **File → New → Terminal**

3. **Paste the same command** from Option A

4. **Test** with curl command above

---

## STEP 2: Expose Port 8000 (2 minutes)

1. **Go back to your pod page**: https://www.runpod.io/console/pods

2. **Find the "HTTP Services" section** (where you see Port 8888)

3. **Click "+ Add Port" or "Edit"** button

4. **Fill in**:
   - Port: `8000`
   - Protocol: `HTTP`  
   - Name: `vLLM API` (optional)

5. **Click Save**

6. **Verify** you now see:
   ```
   Port 8000
   vLLM API
   https://pvj233wwhiu6j3-8000.proxy.runpod.net
   ```

---

## ✅ Test It Works

From your **local machine**, run:

```bash
curl https://pvj233wwhiu6j3-8000.proxy.runpod.net/health
```

Should return: `{"status":"ok"}`

Then test chat:
```bash
curl https://pvj233wwhiu6j3-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

Should return AI response!

---

## 🚀 Final Step: Test Your App

The backend `.env` is already configured with:
```
LLM_API_URL=https://pvj233wwhiu6j3-8000.proxy.runpod.net/v1
```

Now restart your services:

```bash
# Terminal 1 - Backend
cd /Users/omer/Desktop/ai-stanbul/backend
uvicorn main:app --reload
```

```bash
# Terminal 2 - Frontend  
cd /Users/omer/Desktop/ai-stanbul/frontend
npm run dev
```

Then open: http://localhost:5173

**Test the chat!** It should work now! 🎉

---

## 🐛 Still Not Working?

### Check if vLLM is running:
```bash
# Via Web Terminal or Jupyter Lab Terminal
ps aux | grep vllm | grep -v grep
```

### Check vLLM logs:
```bash
tail -100 /root/vllm.log
```

### If you see "CUDA out of memory":
```bash
pkill -9 -f vllm
python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype auto \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.7 \
  > /root/vllm.log 2>&1 &
```

### If port 8000 can't be exposed via UI:
Use SSH tunnel as backup:
```bash
# From your local machine
ssh -L 8000:localhost:8000 pvj233wwhiu6j3-64411542@ssh.runpod.io -i ~/.ssh/id_ed25519
```

Then update backend `.env`:
```
LLM_API_URL=http://localhost:8000/v1
```

---

## 📋 Quick Summary

1. ✅ Start vLLM on port 8000 (inside pod)
2. ✅ Expose port 8000 via RunPod UI
3. ✅ Test with curl
4. ✅ Restart backend & frontend
5. ✅ Test chat in browser

---

## Need Help?

See detailed guides:
- `START_VLLM_ON_RUNPOD.md` - Complete vLLM startup guide
- `RUNPOD_PORT_8000_EXPOSURE.md` - Port exposure details
- `FINAL_DEPLOYMENT_CHECKLIST.md` - Full deployment checklist
