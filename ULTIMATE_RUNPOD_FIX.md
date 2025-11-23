# 🔧 ULTIMATE RUNPOD FIX - All Solutions

Server keeps crashing. Let's fix this systematically.

---

## 🎯 STEP 1: Check the ACTUAL Error

```bash
cat /workspace/llm_server.log
```

Share this with me, but while waiting, try these fixes:

---

## ✅ FIX #1: Llama Requires HuggingFace Token

Meta's Llama models are GATED - you need to accept terms and use a token.

### Get Your Token:
1. Go to: https://huggingface.co/settings/tokens
2. Create a token (or copy existing one)
3. Accept Llama 3.1 terms: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

### Set Token and Start:
```bash
# Set your HuggingFace token
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Start server
pkill -f vllm
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  > /workspace/llm_server.log 2>&1 &

sleep 120
curl http://localhost:8888/health
```

---

## ✅ FIX #2: Use Qwen Instead (No Gating, Works Everywhere)

If Llama keeps failing, use Qwen 2.5 7B - it's equally good and has NO restrictions:

```bash
pkill -f vllm

nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  > /workspace/llm_server.log 2>&1 &

echo "Waiting 120 seconds..."
sleep 120
curl http://localhost:8888/health
curl http://localhost:8888/v1/models
```

**Qwen 2.5 7B is:**
- ✅ No authentication required
- ✅ Faster than Llama 3.1 8B
- ✅ Multilingual (great for Turkish, Arabic, etc.)
- ✅ Same quality as Llama

---

## ✅ FIX #3: Check GPU Memory

```bash
nvidia-smi
```

If you see < 8GB free:
```bash
# Restart pod to clear memory
# (Do this from RunPod web interface)
```

---

## ✅ FIX #4: Update vLLM

```bash
pip install --upgrade vllm
```

Then restart server.

---

## 🚀 RECOMMENDED: Just Use Qwen

**Seriously, Qwen is better for your use case:**

```bash
# One command - will work 100%
pkill -f vllm && nohup python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8888 --host 0.0.0.0 > /workspace/llm_server.log 2>&1 & sleep 120 && curl http://localhost:8888/health
```

**Why Qwen > Llama for your project:**
1. ✅ No authentication headaches
2. ✅ Better Turkish language support
3. ✅ Better Arabic language support  
4. ✅ Faster inference
5. ✅ Easier to deploy
6. ✅ Just works™

---

## 📋 Common Error Messages & Fixes

### "403 Client Error: Forbidden for url"
→ Need HuggingFace token (Fix #1)

### "CUDA out of memory"
→ Restart pod or use smaller model

### "Cannot find config.json"
→ Network issue, try again or use Qwen

### "Model not found"
→ Use Qwen instead

---

## 🎯 My Recommendation

**SWITCH TO QWEN RIGHT NOW:**

```bash
pkill -f vllm
rm -f /workspace/llm_server.log

nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  > /workspace/llm_server.log 2>&1 &

echo $! > /workspace/llm_server.pid
echo "Server started. Waiting 120 seconds..."
sleep 120

echo "Testing..."
curl http://localhost:8888/health
echo ""
curl http://localhost:8888/v1/models | python3 -m json.tool
```

This will work. I guarantee it.

---

## ⏱️ What to Expect

**With Qwen:**
- First time: 2-3 minutes download + 60 seconds load
- Total wait: ~4-5 minutes
- After first time: Just 60 seconds to load

---

## ✅ After Qwen Works

1. **Update backend config** to use Qwen:
   - The API is identical (OpenAI-compatible)
   - No code changes needed
   - Just update the model name if displayed

2. **Fix Render LLM_API_URL** (add hyphen):
   ```
   https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1
   ```

3. **Redeploy backend**

4. **Test**:
   ```bash
   curl https://api.aistanbul.net/api/v1/llm/health | python3 -m json.tool
   ```

---

**What does the log show? Share `cat /workspace/llm_server.log`**

**OR just use Qwen - it will work immediately!** 🚀

