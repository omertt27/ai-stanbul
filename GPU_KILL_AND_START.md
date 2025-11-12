# 🔥 KILL PORT 8888 AND START FRESH

## Step 1: Kill Everything on Port 8888

```bash
lsof -ti:8888 | xargs kill -9 2>/dev/null || echo "No process found on port 8888"
```

**Wait 2 seconds**, then verify:

```bash
lsof -i:8888
```

**Should show:** Nothing (empty output = port is free)

---

## Step 2: Start vLLM Server

```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B \
  --host 0.0.0.0 \
  --port 8888 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  > vllm_server.log 2>&1 &
```

**Wait 5 seconds**, then check the log:

```bash
tail -20 vllm_server.log
```

**Watch for:** `INFO:     Uvicorn running on http://0.0.0.0:8888`

---

## Step 3: Test Health Endpoint

```bash
curl http://localhost:8888/health
```

**Should show:** `{"status":"healthy",...}`

---

## If Port 8888 Still Blocked:

Find and kill ALL Python processes:

```bash
ps aux | grep python | grep vllm
```

Then manually kill each PID:

```bash
kill -9 <PID>
```

Or kill ALL Python processes (⚠️ nuclear option):

```bash
pkill -9 python
```

Then go back to Step 2.

---

**Done!** 🎉
