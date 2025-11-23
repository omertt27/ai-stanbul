# 🚨 RunPod Disk Space Issue - CRITICAL FIX

## Problem
```
OSError: [Errno 122] Disk quota exceeded
```

**Cause:** Llama 3.1 8B model is ~16GB total, but your RunPod instance doesn't have enough disk space in `/workspace/.cache/`

---

## 🔍 Check Current Disk Usage

```bash
# Check disk space
df -h

# Check HuggingFace cache size
du -sh ~/.cache/huggingface/

# Check workspace size
du -sh /workspace/

# Find largest files
du -sh /workspace/* | sort -rh | head -20
```

---

## 🧹 Solution 1: Clean Up Disk Space (Quick Fix)

### Step 1: Remove old/incomplete downloads
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/hub/*

# Check space freed
df -h
```

### Step 2: Remove other large files
```bash
# Find and remove large log files
find /workspace -type f -name "*.log" -size +100M -delete

# Remove temp files
rm -rf /tmp/*

# Remove pip cache
rm -rf ~/.cache/pip/*

# Check space again
df -h
```

---

## 🎯 Solution 2: Use Container Disk Instead (Recommended)

RunPod has two storage locations:
- **Container Disk**: Usually larger, temporary (20-50GB+)
- **Workspace Disk**: Smaller, persistent (5-20GB)

### Change HuggingFace cache location to container disk:

```bash
# Set HuggingFace cache to container disk
export HF_HOME=/root/.cache/huggingface
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Verify location
echo $HF_HOME

# Start server (it will download to container disk now)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 &
```

---

## 🔄 Solution 3: Use Smaller Model (Alternative)

If disk space is severely limited, use a smaller model:

### Option A: Qwen 2.5 7B (Reliable, ~4GB)
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 &
```

### Option B: Llama 3.2 3B (Smallest, ~2GB)
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 &
```

---

## 💾 Solution 4: Upgrade RunPod Instance

If you need Llama 3.1 8B specifically:

1. **Stop current pod**
2. **Create new pod with larger disk**:
   - Go to RunPod dashboard
   - Select template with more disk space (50GB+ container + 20GB+ volume)
   - Common choice: "RunPod PyTorch" with 50GB container

3. **Redeploy with more space**

---

## ✅ Step-by-Step Fix (Recommended)

### 1. Check available space
```bash
df -h
```

Look for available space on `/` (container) vs `/workspace` (volume)

### 2. If container has more space, use it:
```bash
# Clear workspace cache
rm -rf ~/.cache/huggingface/hub/*

# Set cache to container
export HF_HOME=/root/.cache/huggingface
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Start server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 &
```

### 3. If still not enough, use Qwen instead:
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 &
```

---

## 📊 Disk Space Requirements

| Model | Download Size | Runtime Size | Total Needed |
|-------|--------------|--------------|--------------|
| Llama 3.1 8B | ~16GB | ~8GB | **25GB+** |
| Qwen 2.5 7B | ~8GB | ~4GB | **15GB+** |
| Llama 3.2 3B | ~6GB | ~3GB | **10GB+** |

**Recommendation:** Always have 2x the model size free for comfortable operation.

---

## 🔍 Verify After Fix

```bash
# Check disk space
df -h

# Start download and monitor
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code &

# In another terminal, watch disk usage
watch -n 5 'df -h && echo "---" && du -sh ~/.cache/huggingface/ 2>/dev/null || du -sh /root/.cache/huggingface/ 2>/dev/null'
```

---

## 🆘 Quick Decision Tree

1. **Check disk space:** `df -h`
2. **>25GB free on container (/) or workspace?**
   - ✅ YES → Use Llama 3.1 8B with proper cache location
   - ❌ NO → Continue to step 3

3. **>15GB free?**
   - ✅ YES → Use Qwen 2.5 7B (smaller, equally good)
   - ❌ NO → Continue to step 4

4. **>10GB free?**
   - ✅ YES → Use Llama 3.2 3B (smallest)
   - ❌ NO → Clean up disk or upgrade pod

---

## 🚀 Recommended Next Steps

1. **SSH into RunPod** (you're probably already there)
2. **Run:** `df -h` - check what space you have
3. **Choose path:**
   - If 25GB+ free: Clear cache, set HF_HOME, use Llama 3.1 8B
   - If 15-25GB free: Use Qwen 2.5 7B
   - If <15GB free: Upgrade pod or use Llama 3.2 3B

4. **Test the server** after it starts
5. **Update backend if needed** (model name in Render env vars)

---

## 📝 One-Liner Commands

### For Llama 3.1 8B (if space allows):
```bash
rm -rf ~/.cache/huggingface/hub/* && export HF_HOME=/root/.cache/huggingface && export HF_TOKEN="hf_YOUR_TOKEN_HERE" && python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8888 --host 0.0.0.0 --trust-remote-code --max-model-len 4096 &
```

### For Qwen 2.5 7B (smaller, reliable):
```bash
rm -rf ~/.cache/huggingface/hub/* && export HF_TOKEN="hf_YOUR_TOKEN_HERE" && python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8888 --host 0.0.0.0 --trust-remote-code --max-model-len 4096 &
```

---

**Choose based on your available disk space!** 🎯
