# 🔧 RunPod LLM Server Failed to Start

**Error**: `Failed to connect to localhost port 8888`  
**Exit Code**: 2 (indicates an error occurred)

---

## 🔍 What to Do Now (In RunPod SSH)

### Step 1: Check the Error Log

The server tried to start but crashed. Check what went wrong:

```bash
tail -50 /workspace/server.log
```

**Look for these common errors:**

#### Error 1: "Out of CUDA memory" or "CUDA error"
```
CUDA out of memory. Tried to allocate X GB...
```
**Fix**: Your GPU doesn't have enough memory
- Restart the pod to clear memory
- Or use a smaller model/higher quantization

#### Error 2: "No module named 'transformers'" or similar
```
ModuleNotFoundError: No module named 'transformers'
```
**Fix**: Missing dependencies
```bash
pip install transformers accelerate bitsandbytes torch
```

#### Error 3: "File not found" or "No such file or directory"
```
FileNotFoundError: [Errno 2] No such file or directory: '/workspace/models/...'
```
**Fix**: Model files not downloaded or wrong path

#### Error 4: "Port 8888 already in use"
```
OSError: [Errno 98] Address already in use
```
**Fix**: Kill existing process
```bash
pkill -f llm_api_server
# Wait 5 seconds, then restart
```

---

### Step 2: List Available Server Files

Check what Python server files you have:

```bash
ls -lh /workspace/*.py | grep -E "(llm|server|api)"
```

**Common filenames:**
- `llm_api_server_4bit.py`
- `llm_api_server.py`
- `server.py`
- `api_server.py`

---

### Step 3: Check GPU Status

```bash
nvidia-smi
```

**Look for:**
- GPU memory usage (should have several GB free)
- Any existing Python processes using GPU
- GPU temperature (shouldn't be overheating)

---

### Step 4: Try Alternative Startup Methods

#### Method A: Start with full output (see errors immediately)
```bash
cd /workspace
python llm_api_server_4bit.py
```
Don't use `&` - let it run in foreground to see errors

#### Method B: Try different server file (if available)
```bash
# List all Python files
ls *.py

# Try different ones:
python llm_api_server.py
# or
python server.py
```

#### Method C: Start with verbose logging
```bash
python llm_api_server_4bit.py --verbose 2>&1 | tee server.log
```

---

### Step 5: Check Python and PyTorch

```bash
# Check Python version
python --version

# Check if PyTorch can see GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

**If CUDA not available:**
```bash
# Reinstall PyTorch with CUDA support
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118
```

---

## 🎯 Quick Diagnostic Script

I've created a diagnostic script. Copy this to RunPod:

```bash
# On your local machine:
scp diagnose_runpod_server.sh ytc61lal7ag5sy-64410fe8@ssh.runpod.io:/workspace/

# Then on RunPod:
chmod +x /workspace/diagnose_runpod_server.sh
./diagnose_runpod_server.sh
```

---

## 🚨 Most Likely Issues

### Issue #1: GPU Out of Memory (Most Common)
**Symptom**: Server crashes immediately  
**Fix**: Restart pod from RunPod dashboard
```
1. Go to RunPod web interface
2. Stop pod
3. Start pod
4. Wait 30 seconds
5. SSH back in and try again
```

### Issue #2: Wrong Server File
**Symptom**: File not found or import errors  
**Fix**: Find the correct server file
```bash
# Search for server files
find /workspace -name "*server*.py" -o -name "*api*.py"

# Check file contents to identify the right one
head -20 /workspace/llm_api_server_4bit.py
```

### Issue #3: Model Not Loaded
**Symptom**: "Model not found" or similar  
**Fix**: Check model path in server file
```bash
# Check what models are available
ls -lh /workspace/models/

# Check what the server is looking for
grep -n "model" /workspace/llm_api_server_4bit.py | head -10
```

---

## 📋 Step-by-Step Recovery

### If you can't find the issue:

1. **Share the error log:**
   ```bash
   tail -50 /workspace/server.log
   ```
   Copy the output and share it

2. **Check what files you have:**
   ```bash
   ls -lh /workspace/
   ```

3. **Check disk space:**
   ```bash
   df -h /workspace
   ```

4. **Check if model files exist:**
   ```bash
   du -sh /workspace/models/*
   ```

---

## 🔄 Alternative: Use Different Model Server

If the current server won't start, you can use vLLM:

```bash
# Install vLLM
pip install vllm

# Start with a working model
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0
```

---

## 📞 Need the Error Details

To help you further, please share:

```bash
# Run these commands in RunPod and share output:

# 1. Error log
tail -50 /workspace/server.log

# 2. Available files
ls -lh /workspace/*.py

# 3. GPU status
nvidia-smi

# 4. Python/CUDA check
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

**Once you share the error log, I can provide the exact fix!**

