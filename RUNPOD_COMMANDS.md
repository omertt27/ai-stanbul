# 🎉 YOU'RE IN! - RunPod Commands

You're successfully logged into RunPod! Now run these commands:

## Step 1: Check for Llama Model

```bash
find /root/.cache -name "*Llama*3.1*8B*" -type d
```

**If you see a path like:**
```
/root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/...
```
Then the model exists! Continue to Step 2.

**If you see nothing:**
The model needs to be downloaded. Let me know and I'll give you the download commands.

---

## Step 2: Check GPU

```bash
nvidia-smi
```

Should show your GPU (RTX A5000 or similar) with memory available.

---

## Step 3: Start vLLM

Copy and paste this entire block:

```bash
# Find the model snapshot path
MODEL_PATH=$(find /root/.cache/huggingface/hub -name "*Llama-3.1-8B-Instruct" -type d -path "*/snapshots/*" | head -1)

echo "Using model: $MODEL_PATH"

# Kill any old vLLM processes
pkill -9 -f vllm 2>/dev/null || true
sleep 2

# Start vLLM
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 1024 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  > /root/vllm.log 2>&1 &

echo "vLLM starting... Process ID: $!"
echo "Waiting 30 seconds for startup..."
sleep 30

# Test if vLLM is responding
curl http://localhost:8000/v1/models
```

---

## Step 4: Check if vLLM is Working

If you see JSON output with the model name, **SUCCESS!** 🎉

Example output:
```json
{"object":"list","data":[{"id":"meta-llama/Meta-Llama-3.1-8B-Instruct",...}]}
```

---

## Step 5: Exit RunPod

```bash
exit
```

---

## What to Do Next

After exiting RunPod, you'll:
1. Create SSH tunnel from Mac
2. Start backend
3. Start frontend  
4. Test chatbot!

See `FINAL_DEPLOY.md` for the remaining steps.

---

**Start with Step 1 above and tell me what you see!** 🚀
