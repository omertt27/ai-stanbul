# 🔧 vLLM Engine Failed - Troubleshooting Guide

**Error:** `RuntimeError: Engine core initialization failed`

This means vLLM couldn't initialize the GPU engine. Let's diagnose and fix it.

---

## 🔍 Step 1: Check GPU Memory

Run this in your RunPod terminal:

```bash
nvidia-smi
```

**Look for:**
- How much GPU memory is available?
- Is another process using the GPU?
- What GPU model do you have?

---

## 🔍 Step 2: Check Full Error Logs

```bash
# View the last 100 lines of logs
tail -100 /workspace/vllm.log

# Look for error messages about:
# - CUDA out of memory
# - Model file not found
# - Incompatible GPU
```

**Copy and paste the full error here!**

---

## ✅ Solution A: Reduce Memory Usage

The model might be too large for your GPU. Try with **reduced memory settings**:

```bash
# Stop any running vLLM
pkill -f vllm

# Start with lower memory settings
nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.70 \
  --max-model-len 2048 \
  --port 8000 \
  --host 0.0.0.0 \
  --enforce-eager \
  > /workspace/vllm.log 2>&1 &

# Wait 30 seconds, then check logs
sleep 30
tail -50 /workspace/vllm.log
```

**Changes:**
- `--gpu-memory-utilization 0.70` (was 0.85) - Use less GPU memory
- `--max-model-len 2048` (was 4096) - Shorter context window
- `--enforce-eager` - Disable CUDA graphs (uses less memory)

---

## ✅ Solution B: Check Model Files

The model might not be properly downloaded:

```bash
# Check if model directory exists
ls -lh /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/

# You should see files like:
# - config.json
# - tokenizer.json
# - model-00001-of-00002.safetensors (or similar)

# If missing or incomplete, re-download:
export HF_TOKEN=AISTANBUL
export HF_HOME=/workspace/.cache

huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --local-dir /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --cache-dir /workspace/.cache
```

---

## ✅ Solution C: Use Even Smaller Model

If your GPU has limited memory (< 16GB), try a smaller model:

### **Option 1: Llama 3.2 3B (smaller, faster)**

```bash
# Download smaller model (~2GB)
export HF_TOKEN=AISTANBUL
export HF_HOME=/workspace/.cache

huggingface-cli download hugging-quants/Meta-Llama-3.2-3B-Instruct-AWQ-INT4 \
  --local-dir /workspace/Meta-Llama-3.2-3B-Instruct-AWQ-INT4 \
  --cache-dir /workspace/.cache

# Start vLLM with smaller model
pkill -f vllm

nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.2-3B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.70 \
  --max-model-len 2048 \
  --port 8000 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &
```

---

## ✅ Solution D: Check vLLM Version Compatibility

Your vLLM might be too new and have bugs. Try stable version:

```bash
# Install specific stable version
pip install vllm==0.6.1.post2

# Then try starting vLLM again
```

---

## 🔍 Diagnostic Commands

Run these and share the output:

```bash
echo "=== GPU INFO ==="
nvidia-smi

echo -e "\n=== VLLM VERSION ==="
pip show vllm | grep Version

echo -e "\n=== MODEL FILES ==="
ls -lh /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/ 2>/dev/null || echo "Model not found"

echo -e "\n=== DISK SPACE ==="
df -h /workspace

echo -e "\n=== LAST 30 LINES OF ERROR LOG ==="
tail -30 /workspace/vllm.log
```

---

## 🎯 Most Common Fixes (Try in Order)

### **Fix 1: Lower Memory Usage (Easiest)**

```bash
pkill -f vllm

nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.60 \
  --max-model-len 1024 \
  --port 8000 \
  --host 0.0.0.0 \
  --enforce-eager \
  > /workspace/vllm.log 2>&1 &

sleep 30
tail -f /workspace/vllm.log
```

### **Fix 2: Use Different Tensor Parallelism**

```bash
pkill -f vllm

nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.70 \
  --max-model-len 2048 \
  --port 8000 \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  > /workspace/vllm.log 2>&1 &
```

### **Fix 3: Disable AWQ Quantization (Use FP16 instead)**

```bash
pkill -f vllm

# Try without AWQ quantization
nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --dtype float16 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 2048 \
  --port 8000 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &
```

---

## 📊 What GPU Do You Have?

**Check with:**
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
```

**Common RunPod GPUs:**
- **RTX 3090** (24GB) - Should work with 8B model
- **RTX 4090** (24GB) - Should work fine
- **A40** (48GB) - Should work perfectly
- **RTX 3080** (10GB) - Might struggle, use 3B model
- **RTX 3060** (12GB) - Use lower settings

---

## 🆘 If Nothing Works

Try the **smallest possible model** (Phi-2, 2.7B):

```bash
# Download tiny model
huggingface-cli download microsoft/phi-2 \
  --local-dir /workspace/phi-2

# Start vLLM
pkill -f vllm

nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/phi-2 \
  --dtype float16 \
  --gpu-memory-utilization 0.60 \
  --max-model-len 1024 \
  --port 8000 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &
```

---

## 📝 Next Steps

**Please run the diagnostic commands above and share:**
1. GPU model and memory
2. Last 30 lines of `/workspace/vllm.log`
3. Output of `ls -lh /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/`

Then I'll give you the exact command that will work for your setup!

---

**Quick Test:** Try Fix 1 (Lower Memory Usage) first - it solves 80% of these errors!
