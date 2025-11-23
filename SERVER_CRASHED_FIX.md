# 🔍 Quick Diagnosis - Server Crashed

**Status**: Server exited with code 1 (error)

---

## 🎯 Check the Error Log NOW

In your RunPod SSH, run this:

```bash
cat /workspace/llm_server.log
```

Or if it's too long:

```bash
tail -100 /workspace/llm_server.log
```

---

## 🔧 Common Issues & Quick Fixes

### Issue 1: AWQ quantization not supported
**Error in log**: "awq not supported" or "quantization method not found"

**Fix**: Try without quantization or with different method:
```bash
# Option A: No quantization (uses more memory but more compatible)
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --max-model-len 4096 \
  > /workspace/llm_server.log 2>&1 &

# Option B: Use different quantization
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --load-format bitsandbytes \
  --quantization bitsandbytes \
  > /workspace/llm_server.log 2>&1 &
```

---

### Issue 2: CUDA out of memory
**Error in log**: "CUDA out of memory" or "OutOfMemoryError"

**Fix**: Use smaller max length:
```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8 \
  > /workspace/llm_server.log 2>&1 &
```

---

### Issue 3: Model not found / Download failed
**Error in log**: "Model not found" or "Failed to download"

**Fix**: Check if you need HuggingFace token:
```bash
# Set token (if needed)
export HF_TOKEN="your_huggingface_token"

# Then start
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  > /workspace/llm_server.log 2>&1 &
```

---

### Issue 4: vLLM version incompatible
**Error in log**: "ImportError" or version conflicts

**Fix**: Update vLLM:
```bash
pip install --upgrade vllm
```

---

## 🚀 Recommended: Try Without Quantization First

Since AWQ is causing issues, start with basic config:

```bash
# Kill any existing processes
pkill -f vllm

# Start with minimal config (most compatible)
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  > /workspace/llm_server.log 2>&1 &

echo $! > /workspace/llm_server.pid

# Wait and test
sleep 90
curl http://localhost:8888/health
```

This will use automatic settings (vLLM will optimize for your GPU).

---

## 📋 Diagnostic Commands

Run these in RunPod to help diagnose:

```bash
# 1. Check error log
tail -50 /workspace/llm_server.log

# 2. Check GPU memory
nvidia-smi

# 3. Check vLLM version
pip show vllm

# 4. Check Python version
python --version

# 5. Check if port is in use
lsof -i :8888
```

---

## 🎯 Next Steps

1. **Run this NOW**: `tail -50 /workspace/llm_server.log`
2. **Share the error** you see
3. I'll give you the exact fix based on the error

---

## ⚡ Alternative: Use Smaller/Different Model

If Llama 3.1 8B keeps crashing, try a smaller model first:

```bash
# Option 1: Llama 3.2 3B (smaller, faster)
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  > /workspace/llm_server.log 2>&1 &

# Option 2: Qwen 2.5 7B (good performance)
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  > /workspace/llm_server.log 2>&1 &
```

---

**Please share the output of**: `tail -50 /workspace/llm_server.log` 🔍

