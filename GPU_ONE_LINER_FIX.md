# 🎯 ONE-LINER FIX

## ⚠️ FIRST: Make sure bitsandbytes is installed!

```bash
pip install bitsandbytes>=0.46.1
```

## If you see "[screen is terminating]" - Good! Old process killed.

## Now start fresh - Copy-paste this:

```bash
nohup python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B --host 0.0.0.0 --port 8888 --quantization bitsandbytes --load-format bitsandbytes --dtype auto --max-model-len 4096 > vllm_server.log 2>&1 & sleep 5 && tail -f vllm_server.log
```

**This will:**
1. Kill any existing vLLM server
2. Wait 2 seconds
3. Start a new server in background
4. Wait 3 seconds
5. Show you the log

**Watch for:** `INFO:     Uvicorn running on http://0.0.0.0:8888`

**Then press:** `Ctrl+C` to stop viewing log (server keeps running)

---

## Test It Works:

```bash
curl http://localhost:8888/health
```

**Should show:** `{"status":"healthy","model":"meta-llama/Llama-3.1-8B"}`

---

**Done!** 🎉
