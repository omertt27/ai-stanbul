# 🚀 START vLLM RIGHT NOW - Your RunPod

## ✅ Status: vLLM is NOT running (confirmed)

Output from `ps aux | grep vllm`:
```
root  6797  0.0  0.0  3532  1732 pts/1  S+  10:52  0:00 grep --color=auto vllm
```
Only the grep command is running - **vLLM is NOT active**.

## 🎯 STEP-BY-STEP: Start vLLM Now

You're already in RunPod SSH. Follow these commands:

### 1️⃣ Check What Models You Have

```bash
ls -lh /workspace/
```

Look for directories like:
- `Meta-Llama-3.1-8B-Instruct-AWQ-INT4/`
- `Meta-Llama-3-8B-Instruct/`
- `llama-3.1-8b/`
- `models/`

### 2️⃣ Start vLLM (Copy-Paste This)

```bash
cd /workspace

nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --port 8888 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &
```

**Note**: If your model has a different name, replace the `--model` path.

Expected output:
```
[1] 12345
nohup: ignoring input and appending output to 'vllm.log'
```

### 3️⃣ Watch Startup Logs (Takes 2-3 Minutes)

```bash
tail -f /workspace/vllm.log
```

You'll see:
1. `Loading model...`
2. `Loading weights...` ← This takes the longest
3. `Application startup complete` ← DONE! Press Ctrl+C

### 4️⃣ Verify It's Running

```bash
# Check process
ps aux | grep vllm

# Should show python process now (not just grep)

# Check port
curl http://localhost:8888/health

# Should return:
# {"status":"healthy","model":"..."}
```

### 5️⃣ Test from Your Mac

Open a **NEW terminal on your Mac** and test the public URL:

```bash
curl https://4r1su4zfuok0s7-8888.proxy.runpod.net/health
```

Expected: `{"status":"healthy",...}`

### 6️⃣ Update Your .env File

On your Mac (not RunPod):

```bash
cd /Users/omer/Desktop/ai-stanbul
nano .env

# Update these lines:
AI_ISTANBUL_LLM_MODE=runpod
PURE_LLM_MODE=true
LLM_API_URL=https://4r1su4zfuok0s7-8888.proxy.runpod.net

# Save: Ctrl+O, Enter, Ctrl+X
```

### 7️⃣ Test Your Chat

Go to your frontend and try: "hi"

You should get a real AI response, not the fallback error!

## 🔥 Quick Command (One-Liner)

If you want to copy-paste just one command:

```bash
cd /workspace && nohup python -m vllm.entrypoints.openai.api_server --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --quantization awq --dtype half --gpu-memory-utilization 0.85 --max-model-len 2048 --port 8888 --host 0.0.0.0 > /workspace/vllm.log 2>&1 & tail -f /workspace/vllm.log
```

Wait for "Application startup complete", then press Ctrl+C.

## 🛠️ Troubleshooting

### Model Not Found?
```bash
# Find your model
ls -la /workspace/
find /workspace -name "*llama*" -o -name "*Llama*"

# Use the correct path in --model
```

### Python Command Not Found?
```bash
# Try python3
python3 -m vllm.entrypoints.openai.api_server ...
```

### Port 8888 Already in Use?
```bash
# Kill existing process
lsof -i :8888
kill -9 <PID>
```

## ✅ Your Endpoints

Once running:
- **Local (on pod)**: `http://localhost:8888`
- **Public (from Mac)**: `https://4r1su4zfuok0s7-8888.proxy.runpod.net`

## 📝 Summary

1. Run the nohup command above
2. Wait 2-3 minutes for "Application startup complete"
3. Test with `curl http://localhost:8888/health`
4. Update `.env` on your Mac
5. Test chat - should work now!

**Good luck!** 🚀
