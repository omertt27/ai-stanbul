# 🚀 RunPod Installation Guide - Copy/Paste into RunPod Terminal

## ⚠️ IMPORTANT: Before you start

1. **Get HuggingFace Token:**
   - Go to: https://huggingface.co/settings/tokens
   - Create a token with **READ** access
   - Accept Llama 3.1 license: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

2. **Have your token ready!** You'll need it below.

---

## 📋 Copy this ENTIRE script and paste into RunPod terminal:

```bash
# Install everything
apt-get update -qq && \
apt-get install -y git curl && \
pip install --upgrade pip && \
pip install --upgrade huggingface-hub transformers accelerate tokenizers vllm && \
echo "✅ All packages installed!"

# Login to HuggingFace (PASTE YOUR TOKEN when prompted)
echo ""
echo "🔑 Paste your HuggingFace token:"
huggingface-cli login

# Check GPU
echo ""
echo "🎮 GPU Status:"
nvidia-smi

# Start vLLM
echo ""
echo "🚀 Starting vLLM..."
pkill -9 -f vllm 2>/dev/null || true
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.5 \
  --trust-remote-code \
  > /root/vllm.log 2>&1 &

echo $! > /root/vllm.pid
echo "✅ vLLM started! PID: $(cat /root/vllm.pid)"
echo ""
echo "⏳ Waiting 90 seconds for model to load..."
echo "   (First time will download ~16GB model)"
echo ""

# Wait and test
for i in {1..30}; do
  sleep 3
  if [ $((i % 5)) -eq 0 ]; then
    echo "   $((i*3))s elapsed..."
    tail -3 /root/vllm.log | grep -E "ERROR|WARNING|loaded|ready" || true
  fi
  if [ $((i % 3)) -eq 0 ]; then
    curl -s http://localhost:8000/v1/models 2>&1 | grep -q "Meta-Llama" && \
    echo "" && echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" && \
    echo "✅ vLLM IS READY!" && \
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" && \
    curl http://localhost:8000/v1/models | head -20 && \
    echo "" && echo "You can close this terminal now!" && \
    exit 0
  fi
done

echo ""
echo "⚠️  Still loading... Check logs:"
echo "   tail -f /root/vllm.log"
```

---

## 📊 What this does:

1. ✅ Installs system packages
2. ✅ Installs HuggingFace libraries
3. ✅ Installs vLLM
4. ✅ Logs into HuggingFace (you provide token)
5. ✅ Downloads Llama 3.1 8B model (~16GB, first time only)
6. ✅ Starts vLLM server on port 8000
7. ✅ Runs persistently (survives terminal close)

---

## ⏱️ Expected time:

- **First time:** 5-10 minutes (downloading model)
- **After restart:** 1-2 minutes (model already cached)

---

## ✅ When you see "vLLM IS READY!" message:

1. **Close the RunPod terminal** (vLLM keeps running!)
2. **Go to your Mac terminal**
3. **Run:** `./setup_fresh_tunnel.sh`
4. **Then we deploy publicly!** 🎉

---

## 🐛 If something fails:

**Check logs:**
```bash
tail -50 /root/vllm.log
```

**Common issues:**

1. **"Access denied"** → Need to accept Llama license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. **"Out of memory"** → Restart pod with fresh GPU memory
3. **"401 Unauthorized"** → Wrong HuggingFace token, run `huggingface-cli login` again

---

## 🔄 If you need to restart vLLM:

```bash
kill $(cat /root/vllm.pid)
pkill -9 -f vllm
# Then run the vLLM start command again (from the script above)
```

---

**Ready? Copy the script above and paste into RunPod terminal!** 🚀
