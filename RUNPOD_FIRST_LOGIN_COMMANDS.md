# RunPod First Login Commands

**Connect to your RunPod pod first!** 🎉

---

## 🔐 **Step 0: SSH into Your Pod**

**⚠️ IMPORTANT: Run on YOUR LOCAL TERMINAL (Mac), not inside RunPod!**

**Option A: SSH via RunPod Proxy (Recommended)**
```bash
ssh v6fu38ees7z8oj-64411544@ssh.runpod.io -i ~/.ssh/id_ed25519
```

**Option B: Direct TCP Connection (Supports SCP & SFTP)**
```bash
ssh root@69.30.85.244 -p 22005 -i ~/.ssh/id_ed25519
```

**Option C: If SSH key not found, try without `-i` flag:**
```bash
ssh v6fu38ees7z8oj-64411544@ssh.runpod.io
```

Use Option A for general terminal access. Use Option B if you need to transfer files.

Once connected, your prompt looks like: `root@v6fu38ees7z8oj:/#`

---

## 🚀 **Quick Setup (Copy & Paste These Commands)**

### **Step 1: Update System & Install Dependencies**

```bash
# Update pip
pip install --upgrade pip

# Install required packages for LLM server (split into smaller commands)
# Part 1: Core ML libraries
pip install transformers accelerate bitsandbytes

# Part 2: Web server libraries
pip install fastapi uvicorn huggingface_hub

# Verify installations (combined command)
pip list | grep -E "transformers|accelerate|bitsandbytes|fastapi|uvicorn|huggingface"
```

---

### **Step 2: Login to HuggingFace**

**⚠️ IMPORTANT: You need a valid HuggingFace token!**

1. **Get a NEW token from HuggingFace:**
   - Go to: https://huggingface.co/settings/tokens
   - Click **"Create new token"**
   - Name: `runpod-llm`
   - Type: **Read** (for downloading models)
   - Click **"Create token"** and copy it immediately!

2. **Login with your NEW token:**

```bash
# Method 1: Use the new 'hf auth login' command (recommended)
hf auth login --token hf_YOUR_NEW_TOKEN_HERE

# Method 2: If that doesn't work, try legacy command
huggingface-cli login --token hf_YOUR_NEW_TOKEN_HERE

# Method 3: Set environment variable and login
export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE
hf auth login --token $HF_TOKEN

# Verify login (check authentication status)
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"

# Expected output: Your HuggingFace username and account info
# Alternative: Check if you can access a public model
# hf repo info meta-llama/Meta-Llama-3.1-8B-Instruct
```

**⚠️ If you get "401 Unauthorized" or "Invalid token":**
```bash
# The token is invalid or expired. You MUST get a new one!
# 1. Go to: https://huggingface.co/settings/tokens
# 2. Revoke the old token if it exists
# 3. Create a NEW token with Read permissions
# 4. Login again with the NEW token

# Clear any cached invalid tokens:
unset HF_TOKEN
rm -rf ~/.cache/huggingface/token

# Then login with new token:
hf auth login --token hf_YOUR_NEW_TOKEN_HERE
```

**⚠️ If you get "command not found":**
```bash
# Install huggingface_hub package
pip install huggingface_hub

# Or use Python directly to login:
python3 -c "from huggingface_hub import login; login(token='hf_YOUR_NEW_TOKEN_HERE')"

# Verify with Python:
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
```

---

### **Step 3: Check GPU**

```bash
# Verify GPU is available
nvidia-smi

# Should show:
# - GPU name (e.g., RTX 4090)
# - Memory (e.g., 24576 MiB)
# - CUDA version
```

**⚠️ If you get "command not found":**

This means your pod wasn't deployed with GPU access. **This is a CRITICAL issue** - you need a GPU for LLM inference!

**Check if you're using the correct container image:**

1. **Go back to RunPod Console:** https://www.runpod.io/console/pods
2. **Stop your current pod** (to avoid charges)
3. **Deploy a new pod with GPU:**
   - Container Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
   - GPU Type: **RTX 4090** or **RTX A6000** (NOT CPU-only!)
   - Pricing: **Spot** or **On-Demand**
   - Region: EUR-IS-1 or EUR-NO-1

**Verify GPU before continuing:**
```bash
# Check if CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Should output:
# CUDA available: True
# GPU count: 1 (or more)

# If False, you MUST redeploy with GPU!
```

**Alternative check:**
```bash
# Check if GPU drivers are loaded
ls -la /dev/nvidia*

# Should show: /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-modeset

# Check CUDA installation
nvcc --version

# If these don't work, the pod doesn't have GPU access
```

---

### **Step 4: Create LLM Server Script**

#### **Step 4.1: Navigate to Persistent Storage**

```bash
# Navigate to persistent storage (survives pod restarts)
cd /workspace

# Verify you're in the right directory
pwd
# Should output: /workspace
```

#### **Step 4.2: Create the Python Server File**

**Method 1: Download Official Meta Llama 3.1 8B (Recommended - Requires Meta Token)**

```bash
# First, ensure you're logged into HuggingFace with your Meta Llama token
# Verify you're logged in
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
# Should show your username and account info

# Option A: Install hf_transfer for faster downloads (Recommended)
pip install hf_transfer

# Option B: Disable fast transfer if you get errors
# unset HF_HUB_ENABLE_HF_TRANSFER

# Download the official Meta Llama 3.1 8B Instruct model
# This will be quantized to 4-bit automatically by the server
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False

# This downloads ~16GB, but will be loaded in 4-bit (uses ~5GB GPU RAM)
# Download time: ~5-10 minutes depending on connection speed

# Check download progress (run this in another terminal)
watch -n 2 "du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct"

# When complete, verify model files exist
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct/
# Should see: config.json, model.safetensors, tokenizer files, etc.

# ✅ Verify all essential files are present
ls /workspace/models/Meta-Llama-3.1-8B-Instruct/ | grep -E "config.json|tokenizer|model"

# Expected files:
# - config.json (model configuration)
# - tokenizer.json (tokenizer data)
# - tokenizer_config.json (tokenizer configuration)
# - model*.safetensors (model weights - multiple files)
# - special_tokens_map.json
# - generation_config.json

# Check total size (should be ~15-16GB, sometimes up to 30GB with all shards)
du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct
# Expected: ~15G to ~30G (depends on model format and shards)

# Check individual model files
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct/*.safetensors
# Should see model-*.safetensors files

# ✅ If you see all files and the model folder exists, download is complete!
# Note: 30GB is normal for the full model with all shards
echo "✅ Model download complete! Ready to create server script."
```

**⚠️ If you get "401 Unauthorized" or "Repository not found":**

This means you need to:
1. **Accept Meta's License Agreement:**
   - Go to: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
   - Click **"Access Llama 3.1 on Hugging Face"**
   - Fill out the form and accept the license
   - Wait for approval (usually instant)

2. **Use a token with proper permissions:**
   ```bash
   # Your token must have READ access
   # Login again if needed
   hf auth login --token hf_YOUR_META_LLAMA_TOKEN_HERE
   
   # Verify access (use Python, as 'hf whoami' doesn't exist)
   python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
   ```

**Method 2: Create Server Script (Official Meta Model with 4-bit Quantization)**

**⚠️ IMPORTANT: Make sure you're in /workspace before creating the file!**

```bash
# Navigate to /workspace first
cd /workspace
pwd
# Should show: /workspace
```

**Step-by-step creation (Copy each command separately):**

```bash
# Part 1: Create file with imports and app setup
cat > /workspace/llm_server.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn
import os

app = FastAPI()
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/Meta-Llama-3.1-8B-Instruct")
EOF
```

```bash
# Part 2: Add quantization configuration
cat >> /workspace/llm_server.py << 'EOF'

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
EOF
```

```bash
# Part 3: Add model loading code
cat >> /workspace/llm_server.py << 'EOF'

print(f"Loading official Meta Llama model from: {MODEL_PATH}...")
print("This may take 2-3 minutes on first load...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
print("Model loaded successfully!")
EOF
```

```bash
# Part 4: Add request model
cat >> /workspace/llm_server.py << 'EOF'

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 250
EOF
```

```bash
# Part 5: Add health endpoint
cat >> /workspace/llm_server.py << 'EOF'

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "quantization": "4-bit"
    }
EOF
```

```bash
# Part 6: Add generate endpoint
cat >> /workspace/llm_server.py << 'EOF'

@app.post("/generate")
async def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(request.prompt):].strip()
    return {"generated_text": response}
EOF
```

```bash
# Part 7: Add main block
cat >> /workspace/llm_server.py << 'EOF'

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
EOF
```

```bash
# Verify the complete file was created in /workspace
ls -lh /workspace/llm_server.py
cat /workspace/llm_server.py | wc -l
# Should show ~50-60 lines (59 lines is perfect!)

# Quick preview of the file structure
head -5 /workspace/llm_server.py && echo "..." && tail -5 /workspace/llm_server.py
# Should show imports at top and uvicorn.run at bottom

echo "✅ Server script created successfully in /workspace!"
```

**Method 3: Alternative - Pre-quantized GGUF (Faster Download, No Meta Token)**

```bash
# If you prefer a pre-quantized version (smaller download, faster setup)
# Download the quantized Llama 3.1 8B model (4-bit GGUF format)
huggingface-cli download \
  bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --local-dir /workspace/models \
  --local-dir-use-symlinks False

# Create server for GGUF model
cat > llm_server.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn
import os

app = FastAPI()
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

print(f"Loading model from: {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=-1,  # Use all GPU layers
    verbose=False
)
print("Model loaded successfully!")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 250

@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_PATH}

@app.post("/generate")
async def generate(request: GenerateRequest):
    response = llm(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=0.7,
        top_p=0.9,
        echo=False
    )
    return {"generated_text": response["choices"][0]["text"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
EOF

# Install llama-cpp-python for GGUF support
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

**Method 4: Create in Parts (Step-by-step - For Manual Control)**

```bash
# Part 1: Imports and setup
cat > llm_server.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn
import os

app = FastAPI()
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B")

if MODEL_NAME.startswith("https://huggingface.co/"):
    MODEL_NAME = MODEL_NAME.replace("https://huggingface.co/", "")
EOF

# Part 2: Model configuration
cat >> llm_server.py << 'EOF'

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
EOF

# Part 3: Load model
cat >> llm_server.py << 'EOF'

print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
print("Model loaded successfully!")
EOF

# Part 4: Request model
cat >> llm_server.py << 'EOF'

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 250
EOF

# Part 5: Health endpoint
cat >> llm_server.py << 'EOF'

@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_NAME}
EOF

# Part 6: Generate endpoint
cat >> llm_server.py << 'EOF'

@app.post("/generate")
async def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": response}
EOF

# Part 7: Main block
cat >> llm_server.py << 'EOF'

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
EOF
```

**Method 3: Use nano editor (Interactive)**

```bash
nano llm_server.py
# Paste code, then: Ctrl+X, Y, Enter
```

**Method 4: Use vi editor (Advanced)**

```bash
vi llm_server.py
# Press 'i', paste code, Esc, :wq, Enter
```

#### **Step 4.3: Verify File Creation**

```bash
# Check if file exists and its size
ls -lh llm_server.py

# Expected output:
# -rw-r--r-- 1 root root 1.5K Nov 25 12:00 llm_server.py

# View first 20 lines to confirm content
head -20 llm_server.py

# Should show Python imports and FastAPI setup
```

#### **Step 4.4: Validate Python Syntax (Optional but Recommended)**

```bash
# Check for syntax errors
python3 -m py_compile llm_server.py

# If no output, syntax is valid!
# If there's an error, you'll see a SyntaxError message

# Alternative: Run a quick syntax check
python3 -c "import ast; ast.parse(open('llm_server.py').read())"

# No output = success
```

#### **Step 4.5: Review Server Configuration**

```bash
# View full file to understand what it does
cat llm_server.py

# Key components:
# - FastAPI web server
# - Llama 3.1 8B model (4-bit quantized)
# - /health endpoint (check server status)
# - /generate endpoint (generate text from prompts)
# - Runs on port 8888
```

**What this server does:**
- ✅ Loads Meta's Llama 3.1 8B model in 4-bit (saves GPU memory)
- ✅ Creates REST API with 2 endpoints
- ✅ Health check: `GET /health`
- ✅ Text generation: `POST /generate`
- ✅ Runs on port 8888 (accessible via RunPod proxy)

---

### **Step 5: Start the LLM Server (Background)**

```bash
# Install screen for background process management
apt-get update && apt-get install -y screen

# Start server in screen session (runs in background)
screen -S llm-server

# Inside screen session, start the server:
cd /workspace
python llm_server.py

# When you see "Model loaded successfully!" and "Uvicorn running on http://0.0.0.0:8888"
# Press: Ctrl+A, then D (to detach from screen)

# To reattach later:
# screen -r llm-server
```

**Alternative: Run in Foreground (for testing)**
```bash
# First, verify you're in the correct directory and file exists
cd /workspace
pwd
ls -lh llm_server.py

# If file doesn't exist, check where it is
find /workspace -name "llm_server.py" 2>/dev/null

# If file is found elsewhere, either:
# Option 1: Move to /workspace
# mv /path/to/llm_server.py /workspace/

# Option 2: cd to where the file is
# cd /path/where/file/is

# Once you confirm the file exists in current directory, run:
python llm_server.py

# This will show real-time logs
# Press Ctrl+C to stop
```

---

### **Step 6: Test the Server Locally**

```bash
# In a new terminal (or after detaching from screen), test:
curl http://localhost:8888/health

# Expected output:
# {"status":"healthy","model":"meta-llama/Llama-3.1-8B"}

# Test generation
curl -X POST http://localhost:8888/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Istanbul?", "max_tokens": 100}'

# Should return generated text about Istanbul
```

---

### **Step 7: Get Your Public Endpoint**

Your RunPod pod has a public endpoint accessible from anywhere:

1. **Go to RunPod Console:** https://www.runpod.io/console/pods
2. Click on your running pod
3. Click **"Connect"** tab
4. Look for **"HTTP Ports"** section
5. Copy the URL for port **8888**

**Format:**
```
https://<pod-id>-8888.proxy.runpod.net
```

**Example:**
```
https://9575d6f75e8a-8888.proxy.runpod.net
```

---

### **Step 8: Test Public Endpoint**

```bash
# From your local machine (not inside RunPod pod):
curl https://<your-pod-id>-8888.proxy.runpod.net/health

# Test generation from anywhere:
curl -X POST https://<your-pod-id>-8888.proxy.runpod.net/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are the best places to visit in Istanbul?", "max_tokens": 150}'
```

---

## 🔧 **Useful Commands Inside RunPod**

### **Monitor GPU Usage**
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Press Ctrl+C to exit
```

### **Check Disk Space**
```bash
# Container disk (temporary)
df -h /

# Volume disk (persistent)
df -h /workspace
```

### **View Server Logs**
```bash
# If running in screen
screen -r llm-server

# If using systemd (see below)
journalctl -u llm-server -f
```

### **Check Running Processes**
```bash
# Check if server is running
ps aux | grep python

# Check ports
netstat -tulpn | grep 8888
```

---

## 🛡️ **Production Setup (Keep Server Running 24/7)**

### **Option 1: Screen (Simple)**
```bash
# Start
screen -S llm-server
python /workspace/llm_server.py

# Detach: Ctrl+A, D
# Reattach: screen -r llm-server
```

### **Option 2: Systemd Service (Recommended)**
```bash
# Create service file
cat > /etc/systemd/system/llm-server.service << 'EOF'
[Unit]
Description=LLM API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace
Environment="HF_TOKEN=hf_xxxxxxxxxxxxx"
Environment="MODEL_NAME=meta-llama/Llama-3.1-8B"
ExecStart=/usr/bin/python3 /workspace/llm_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

# Enable service (start on boot)
systemctl enable llm-server

# Start service
systemctl start llm-server

# Check status
systemctl status llm-server

# View logs
journalctl -u llm-server -f
```

---

## 📊 **Monitoring & Debugging**

### **Check Model Download Progress**
```bash
# Monitor model cache
du -sh /workspace/models/
watch -n 5 "du -sh /workspace/models/*"
```

### **Memory Usage**
```bash
# RAM usage
free -h

# GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### **Server Health**
```bash
# Check if FastAPI is responding
curl -v http://localhost:8888/health

# Check response time
time curl http://localhost:8888/health
```

---

## 🆘 **Troubleshooting**

### **Issue: "hf_transfer package is not available" during model download**
```bash
# Error: ValueError: Fast download using 'hf_transfer' is enabled but 'hf_transfer' package is not available

# Solution 1: Install hf_transfer (Recommended - much faster downloads)
pip install hf_transfer

# Then retry the download command
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False

# Solution 2: Disable fast transfer (slower but works without hf_transfer)
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0

# Then retry the download
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False
```

### **Issue: "Invalid user token" or "401 Unauthorized"**
```bash
# This means your HuggingFace token is invalid, expired, or has wrong permissions

# Step 1: Get a NEW token
# Go to: https://huggingface.co/settings/tokens
# Create new token with READ permissions

# Step 2: Clear old cached tokens
unset HF_TOKEN
rm -rf ~/.cache/huggingface/token

# Step 3: Login with NEW token
hf auth login --token hf_YOUR_NEW_TOKEN_HERE

# Step 4: Verify it works
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
# Should show your username and account info, NOT an error

# Step 5: Set environment variable for future use
export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE
echo "export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE" >> ~/.bashrc
```

### **Issue: "HF_TOKEN not found"**
```bash
# Set it manually
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Or login directly
hf auth login --token hf_xxxxxxxxxxxxxxxxxxxxx
```

### **Issue: "Out of memory"**
```bash
# Check GPU memory
nvidia-smi

# Reduce max_tokens in generation
# Edit llm_server.py and change default max_tokens from 250 to 150
```

### **Issue: "Model download is slow"**
```bash
# First download is slow (15GB model)
# Check progress:
ls -lh /workspace/models/

# Subsequent runs are instant (model is cached)
```

### **Issue: "Port 8888 already in use"**
```bash
# Check what's using it
lsof -i :8888

# Kill the process
kill -9 <PID>

# Or use different port
python llm_server.py --port 8889
```

### **Issue: "AttributeError: 'dict' object has no attribute 'model_type'" when starting server**

**This error occurs when model files are incomplete or corrupted during download.**

#### **Quick Fix - Check Model Files First:**

```bash
# Step 1: Check if config.json exists and is valid
cd /workspace/models/Meta-Llama-3.1-8B-Instruct
ls -lh config.json

# Step 2: Try to load the config with Python
python3 << 'EOF'
import json
try:
    with open('/workspace/models/Meta-Llama-3.1-8B-Instruct/config.json', 'r') as f:
        config = json.load(f)
    print("✅ Config file is valid JSON")
    print(f"Model type: {config.get('model_type', 'NOT FOUND')}")
    print(f"Architecture: {config.get('architectures', 'NOT FOUND')}")
except Exception as e:
    print(f"❌ Config file is corrupted: {e}")
EOF

# Step 3: Check if all essential model files exist
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct/ | grep -E "config.json|tokenizer|model.*safetensors"

# Expected files:
# - config.json (should be ~1-2KB)
# - tokenizer.json (should be ~1-2MB)
# - tokenizer_config.json
# - model-*.safetensors (should be multiple GB each)

# Step 4: Check total size and file count
du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct
ls /workspace/models/Meta-Llama-3.1-8B-Instruct | wc -l

# Expected: ~15-30GB total, at least 10-15 files
```

#### **If Config is Corrupted or Incomplete - Full Fix:**

```bash
# Step 1: Backup the corrupted model (if needed)
cd /workspace/models
mv Meta-Llama-3.1-8B-Instruct Meta-Llama-3.1-8B-Instruct.backup

# Step 2: Create fresh directory
mkdir -p /workspace/models/Meta-Llama-3.1-8B-Instruct

# Step 3: Verify you're logged into HuggingFace
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
# Should show your username, NOT an error

# Step 4: Install hf_transfer for reliable downloads (optional but recommended)
pip install hf_transfer

# Step 5: Re-download the model (this is the most reliable method)
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False \
  --resume-download

# The --resume-download flag will skip files that are already complete

# Step 6: Monitor download progress (in another terminal)
watch -n 5 "du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct && echo '---' && ls /workspace/models/Meta-Llama-3.1-8B-Instruct | wc -l"

# Step 7: After download completes, verify config.json
python3 << 'EOF'
import json
from transformers import AutoConfig

# Method 1: Check raw JSON
with open('/workspace/models/Meta-Llama-3.1-8B-Instruct/config.json', 'r') as f:
    config = json.load(f)
    print("✅ Config JSON is valid")
    print(f"   Model type: {config.get('model_type')}")
    print(f"   Architecture: {config.get('architectures')}")

# Method 2: Load with transformers (this is what the server uses)
try:
    config = AutoConfig.from_pretrained('/workspace/models/Meta-Llama-3.1-8B-Instruct', trust_remote_code=True)
    print("✅ Config loads correctly with transformers")
    print(f"   Model type: {config.model_type}")
except Exception as e:
    print(f"❌ Failed to load with transformers: {e}")
EOF

# Step 8: Verify all model shards are present
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct/*.safetensors

# Should see multiple files like:
# model-00001-of-00004.safetensors
# model-00002-of-00004.safetensors
# etc.

# Step 9: Once verified, recreate the server script (see next section)
```

#### **Alternative: If Download Keeps Failing - Download Individual Files:**

```bash
# Sometimes batch download fails, so download critical files individually

# Step 1: Create directory
mkdir -p /workspace/models/Meta-Llama-3.1-8B-Instruct
cd /workspace/models/Meta-Llama-3.1-8B-Instruct

# Step 2: Download config files first
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct generation_config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct tokenizer_config.json --local-dir . --local-dir-use-symlinks False

# Step 3: Download tokenizer files
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct tokenizer.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct special_tokens_map.json --local-dir . --local-dir-use-symlinks False

# Step 4: Verify config works before downloading large model files
python3 -c "from transformers import AutoConfig; config = AutoConfig.from_pretrained('.', trust_remote_code=True); print(f'✅ Config OK: {config.model_type}')"

# Step 5: Only if config works, download model weights (large files)
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --include "*.safetensors" --local-dir . --local-dir-use-symlinks False

# Step 6: Final verification
ls -lh
# Should see all files listed above
```

#### **After Model Files are Fixed - Recreate Server Script:**

```bash
# Remove old server file
rm -f /workspace/llm_server.py

# Create new server with all fixes
cat > /workspace/llm_server.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn
import os

app = FastAPI()
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/Meta-Llama-3.1-8B-Instruct")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print(f"Loading official Meta Llama model from: {MODEL_PATH}...")
print("This may take 2-3 minutes on first load...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
print("Model loaded successfully!")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 250

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "quantization": "4-bit"
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(request.prompt):].strip()
    return {"generated_text": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
EOF

# Verify file was created
ls -lh /workspace/llm_server.py
cat /workspace/llm_server.py | wc -l
# Should show ~59 lines

echo "✅ Server script created successfully!"
```

---

## 🔧 **Useful Commands Inside RunPod**

### **Monitor GPU Usage**
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Press Ctrl+C to exit
```

### **Check Disk Space**
```bash
# Container disk (temporary)
df -h /

# Volume disk (persistent)
df -h /workspace
```

### **View Server Logs**
```bash
# If running in screen
screen -r llm-server

# If using systemd (see below)
journalctl -u llm-server -f
```

### **Check Running Processes**
```bash
# Check if server is running
ps aux | grep python

# Check ports
netstat -tulpn | grep 8888
```

---

## 🛡️ **Production Setup (Keep Server Running 24/7)**

### **Option 1: Screen (Simple)**
```bash
# Start
screen -S llm-server
python /workspace/llm_server.py

# Detach: Ctrl+A, D
# Reattach: screen -r llm-server
```

### **Option 2: Systemd Service (Recommended)**
```bash
# Create service file
cat > /etc/systemd/system/llm-server.service << 'EOF'
[Unit]
Description=LLM API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace
Environment="HF_TOKEN=hf_xxxxxxxxxxxxx"
Environment="MODEL_NAME=meta-llama/Llama-3.1-8B"
ExecStart=/usr/bin/python3 /workspace/llm_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

# Enable service (start on boot)
systemctl enable llm-server

# Start service
systemctl start llm-server

# Check status
systemctl status llm-server

# View logs
journalctl -u llm-server -f
```

---

## 📊 **Monitoring & Debugging**

### **Check Model Download Progress**
```bash
# Monitor model cache
du -sh /workspace/models/
watch -n 5 "du -sh /workspace/models/*"
```

### **Memory Usage**
```bash
# RAM usage
free -h

# GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### **Server Health**
```bash
# Check if FastAPI is responding
curl -v http://localhost:8888/health

# Check response time
time curl http://localhost:8888/health
```

---

## 🆘 **Troubleshooting**

### **Issue: "hf_transfer package is not available" during model download**
```bash
# Error: ValueError: Fast download using 'hf_transfer' is enabled but 'hf_transfer' package is not available

# Solution 1: Install hf_transfer (Recommended - much faster downloads)
pip install hf_transfer

# Then retry the download command
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False

# Solution 2: Disable fast transfer (slower but works without hf_transfer)
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0

# Then retry the download
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False
```

### **Issue: "Invalid user token" or "401 Unauthorized"**
```bash
# This means your HuggingFace token is invalid, expired, or has wrong permissions

# Step 1: Get a NEW token
# Go to: https://huggingface.co/settings/tokens
# Create new token with READ permissions

# Step 2: Clear old cached tokens
unset HF_TOKEN
rm -rf ~/.cache/huggingface/token

# Step 3: Login with NEW token
hf auth login --token hf_YOUR_NEW_TOKEN_HERE

# Step 4: Verify it works
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
# Should show your username and account info, NOT an error

# Step 5: Set environment variable for future use
export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE
echo "export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE" >> ~/.bashrc
```

### **Issue: "HF_TOKEN not found"**
```bash
# Set it manually
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Or login directly
hf auth login --token hf_xxxxxxxxxxxxxxxxxxxxx
```

### **Issue: "Out of memory"**
```bash
# Check GPU memory
nvidia-smi

# Reduce max_tokens in generation
# Edit llm_server.py and change default max_tokens from 250 to 150
```

### **Issue: "Model download is slow"**
```bash
# First download is slow (15GB model)
# Check progress:
ls -lh /workspace/models/

# Subsequent runs are instant (model is cached)
```

### **Issue: "Port 8888 already in use"**
```bash
# Check what's using it
lsof -i :8888

# Kill the process
kill -9 <PID>

# Or use different port
python llm_server.py --port 8889
```

### **Issue: "AttributeError: 'dict' object has no attribute 'model_type'" when starting server**

**This error occurs when model files are incomplete or corrupted during download.**

#### **Quick Fix - Check Model Files First:**

```bash
# Step 1: Check if config.json exists and is valid
cd /workspace/models/Meta-Llama-3.1-8B-Instruct
ls -lh config.json

# Step 2: Try to load the config with Python
python3 << 'EOF'
import json
try:
    with open('/workspace/models/Meta-Llama-3.1-8B-Instruct/config.json', 'r') as f:
        config = json.load(f)
    print("✅ Config file is valid JSON")
    print(f"Model type: {config.get('model_type', 'NOT FOUND')}")
    print(f"Architecture: {config.get('architectures', 'NOT FOUND')}")
except Exception as e:
    print(f"❌ Config file is corrupted: {e}")
EOF

# Step 3: Check if all essential model files exist
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct/ | grep -E "config.json|tokenizer|model.*safetensors"

# Expected files:
# - config.json (should be ~1-2KB)
# - tokenizer.json (should be ~1-2MB)
# - tokenizer_config.json
# - model-*.safetensors (should be multiple GB each)

# Step 4: Check total size and file count
du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct
ls /workspace/models/Meta-Llama-3.1-8B-Instruct | wc -l

# Expected: ~15-30GB total, at least 10-15 files
```

#### **If Config is Corrupted or Incomplete - Full Fix:**

```bash
# Step 1: Backup the corrupted model (if needed)
cd /workspace/models
mv Meta-Llama-3.1-8B-Instruct Meta-Llama-3.1-8B-Instruct.backup

# Step 2: Create fresh directory
mkdir -p /workspace/models/Meta-Llama-3.1-8B-Instruct

# Step 3: Verify you're logged into HuggingFace
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
# Should show your username, NOT an error

# Step 4: Install hf_transfer for reliable downloads (optional but recommended)
pip install hf_transfer

# Step 5: Re-download the model (this is the most reliable method)
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False \
  --resume-download

# The --resume-download flag will skip files that are already complete

# Step 6: Monitor download progress (in another terminal)
watch -n 5 "du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct && echo '---' && ls /workspace/models/Meta-Llama-3.1-8B-Instruct | wc -l"

# Step 7: After download completes, verify config.json
python3 << 'EOF'
import json
from transformers import AutoConfig

# Method 1: Check raw JSON
with open('/workspace/models/Meta-Llama-3.1-8B-Instruct/config.json', 'r') as f:
    config = json.load(f)
    print("✅ Config JSON is valid")
    print(f"   Model type: {config.get('model_type')}")
    print(f"   Architecture: {config.get('architectures')}")

# Method 2: Load with transformers (this is what the server uses)
try:
    config = AutoConfig.from_pretrained('/workspace/models/Meta-Llama-3.1-8B-Instruct', trust_remote_code=True)
    print("✅ Config loads correctly with transformers")
    print(f"   Model type: {config.model_type}")
except Exception as e:
    print(f"❌ Failed to load with transformers: {e}")
EOF

# Step 8: Verify all model shards are present
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct/*.safetensors

# Should see multiple files like:
# model-00001-of-00004.safetensors
# model-00002-of-00004.safetensors
# etc.

# Step 9: Once verified, recreate the server script (see next section)
```

#### **Alternative: If Download Keeps Failing - Download Individual Files:**

```bash
# Sometimes batch download fails, so download critical files individually

# Step 1: Create directory
mkdir -p /workspace/models/Meta-Llama-3.1-8B-Instruct
cd /workspace/models/Meta-Llama-3.1-8B-Instruct

# Step 2: Download config files first
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct generation_config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct tokenizer_config.json --local-dir . --local-dir-use-symlinks False

# Step 3: Download tokenizer files
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct tokenizer.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct special_tokens_map.json --local-dir . --local-dir-use-symlinks False

# Step 4: Verify config works before downloading large model files
python3 -c "from transformers import AutoConfig; config = AutoConfig.from_pretrained('.', trust_remote_code=True); print(f'✅ Config OK: {config.model_type}')"

# Step 5: Only if config works, download model weights (large files)
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --include "*.safetensors" --local-dir . --local-dir-use-symlinks False

# Step 6: Final verification
ls -lh
# Should see all files listed above
```

#### **After Model Files are Fixed - Recreate Server Script:**

```bash
# Remove old server file
rm -f /workspace/llm_server.py

# Create new server with all fixes
cat > /workspace/llm_server.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn
import os

app = FastAPI()
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/Meta-Llama-3.1-8B-Instruct")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print(f"Loading official Meta Llama model from: {MODEL_PATH}...")
print("This may take 2-3 minutes on first load...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
print("Model loaded successfully!")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 250

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "quantization": "4-bit"
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(request.prompt):].strip()
    return {"generated_text": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
EOF

# Verify file was created
ls -lh /workspace/llm_server.py
cat /workspace/llm_server.py | wc -l
# Should show ~59 lines

echo "✅ Server script created successfully!"
```

---

## 🔧 **Useful Commands Inside RunPod**

### **Monitor GPU Usage**
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Press Ctrl+C to exit
```

### **Check Disk Space**
```bash
# Container disk (temporary)
df -h /

# Volume disk (persistent)
df -h /workspace
```

### **View Server Logs**
```bash
# If running in screen
screen -r llm-server

# If using systemd (see below)
journalctl -u llm-server -f
```

### **Check Running Processes**
```bash
# Check if server is running
ps aux | grep python

# Check ports
netstat -tulpn | grep 8888
```

---

## 🛡️ **Production Setup (Keep Server Running 24/7)**

### **Option 1: Screen (Simple)**
```bash
# Start
screen -S llm-server
python /workspace/llm_server.py

# Detach: Ctrl+A, D
# Reattach: screen -r llm-server
```

### **Option 2: Systemd Service (Recommended)**
```bash
# Create service file
cat > /etc/systemd/system/llm-server.service << 'EOF'
[Unit]
Description=LLM API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace
Environment="HF_TOKEN=hf_xxxxxxxxxxxxx"
Environment="MODEL_NAME=meta-llama/Llama-3.1-8B"
ExecStart=/usr/bin/python3 /workspace/llm_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

# Enable service (start on boot)
systemctl enable llm-server

# Start service
systemctl start llm-server

# Check status
systemctl status llm-server

# View logs
journalctl -u llm-server -f
```

---

## 📊 **Monitoring & Debugging**

### **Check Model Download Progress**
```bash
# Monitor model cache
du -sh /workspace/models/
watch -n 5 "du -sh /workspace/models/*"
```

### **Memory Usage**
```bash
# RAM usage
free -h

# GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### **Server Health**
```bash
# Check if FastAPI is responding
curl -v http://localhost:8888/health

# Check response time
time curl http://localhost:8888/health
```

---

## 🆘 **Troubleshooting**

### **Issue: "hf_transfer package is not available" during model download**
```bash
# Error: ValueError: Fast download using 'hf_transfer' is enabled but 'hf_transfer' package is not available

# Solution 1: Install hf_transfer (Recommended - much faster downloads)
pip install hf_transfer

# Then retry the download command
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False

# Solution 2: Disable fast transfer (slower but works without hf_transfer)
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0

# Then retry the download
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False
```

### **Issue: "Invalid user token" or "401 Unauthorized"**
```bash
# This means your HuggingFace token is invalid, expired, or has wrong permissions

# Step 1: Get a NEW token
# Go to: https://huggingface.co/settings/tokens
# Create new token with READ permissions

# Step 2: Clear old cached tokens
unset HF_TOKEN
rm -rf ~/.cache/huggingface/token

# Step 3: Login with NEW token
hf auth login --token hf_YOUR_NEW_TOKEN_HERE

# Step 4: Verify it works
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
# Should show your username and account info, NOT an error

# Step 5: Set environment variable for future use
export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE
echo "export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE" >> ~/.bashrc
```

### **Issue: "HF_TOKEN not found"**
```bash
# Set it manually
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Or login directly
hf auth login --token hf_xxxxxxxxxxxxxxxxxxxxx
```

### **Issue: "Out of memory"**
```bash
# Check GPU memory
nvidia-smi

# Reduce max_tokens in generation
# Edit llm_server.py and change default max_tokens from 250 to 150
```

### **Issue: "Model download is slow"**
```bash
# First download is slow (15GB model)
# Check progress:
ls -lh /workspace/models/

# Subsequent runs are instant (model is cached)
```

### **Issue: "Port 8888 already in use"**
```bash
# Check what's using it
lsof -i :8888

# Kill the process
kill -9 <PID>

# Or use different port
python llm_server.py --port 8889
```

### **Issue: "AttributeError: 'dict' object has no attribute 'model_type'" when starting server**

**This error occurs when model files are incomplete or corrupted during download.**

#### **Quick Fix - Check Model Files First:**

```bash
# Step 1: Check if config.json exists and is valid
cd /workspace/models/Meta-Llama-3.1-8B-Instruct
ls -lh config.json

# Step 2: Try to load the config with Python
python3 << 'EOF'
import json
try:
    with open('/workspace/models/Meta-Llama-3.1-8B-Instruct/config.json', 'r') as f:
        config = json.load(f)
    print("✅ Config file is valid JSON")
    print(f"Model type: {config.get('model_type', 'NOT FOUND')}")
    print(f"Architecture: {config.get('architectures', 'NOT FOUND')}")
except Exception as e:
    print(f"❌ Config file is corrupted: {e}")
EOF

# Step 3: Check if all essential model files exist
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct/ | grep -E "config.json|tokenizer|model.*safetensors"

# Expected files:
# - config.json (should be ~1-2KB)
# - tokenizer.json (should be ~1-2MB)
# - tokenizer_config.json
# - model-*.safetensors (should be multiple GB each)

# Step 4: Check total size and file count
du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct
ls /workspace/models/Meta-Llama-3.1-8B-Instruct | wc -l

# Expected: ~15-30GB total, at least 10-15 files
```

#### **If Config is Corrupted or Incomplete - Full Fix:**

```bash
# Step 1: Backup the corrupted model (if needed)
cd /workspace/models
mv Meta-Llama-3.1-8B-Instruct Meta-Llama-3.1-8B-Instruct.backup

# Step 2: Create fresh directory
mkdir -p /workspace/models/Meta-Llama-3.1-8B-Instruct

# Step 3: Verify you're logged into HuggingFace
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
# Should show your username, NOT an error

# Step 4: Install hf_transfer for reliable downloads (optional but recommended)
pip install hf_transfer

# Step 5: Re-download the model (this is the most reliable method)
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False \
  --resume-download

# The --resume-download flag will skip files that are already complete

# Step 6: Monitor download progress (in another terminal)
watch -n 5 "du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct && echo '---' && ls /workspace/models/Meta-Llama-3.1-8B-Instruct | wc -l"

# Step 7: After download completes, verify config.json
python3 << 'EOF'
import json
from transformers import AutoConfig

# Method 1: Check raw JSON
with open('/workspace/models/Meta-Llama-3.1-8B-Instruct/config.json', 'r') as f:
    config = json.load(f)
    print("✅ Config JSON is valid")
    print(f"   Model type: {config.get('model_type')}")
    print(f"   Architecture: {config.get('architectures')}")

# Method 2: Load with transformers (this is what the server uses)
try:
    config = AutoConfig.from_pretrained('/workspace/models/Meta-Llama-3.1-8B-Instruct', trust_remote_code=True)
    print("✅ Config loads correctly with transformers")
    print(f"   Model type: {config.model_type}")
except Exception as e:
    print(f"❌ Failed to load with transformers: {e}")
EOF

# Step 8: Verify all model shards are present
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct/*.safetensors

# Should see multiple files like:
# model-00001-of-00004.safetensors
# model-00002-of-00004.safetensors
# etc.

# Step 9: Once verified, recreate the server script (see next section)
```

#### **Alternative: If Download Keeps Failing - Download Individual Files:**

```bash
# Sometimes batch download fails, so download critical files individually

# Step 1: Create directory
mkdir -p /workspace/models/Meta-Llama-3.1-8B-Instruct
cd /workspace/models/Meta-Llama-3.1-8B-Instruct

# Step 2: Download config files first
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct generation_config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct tokenizer_config.json --local-dir . --local-dir-use-symlinks False

# Step 3: Download tokenizer files
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct tokenizer.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct special_tokens_map.json --local-dir . --local-dir-use-symlinks False

# Step 4: Verify config works before downloading large model files
python3 -c "from transformers import AutoConfig; config = AutoConfig.from_pretrained('.', trust_remote_code=True); print(f'✅ Config OK: {config.model_type}')"

# Step 5: Only if config works, download model weights (large files)
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --include "*.safetensors" --local-dir . --local-dir-use-symlinks False

# Step 6: Final verification
ls -lh
# Should see all files listed above
```

#### **After Model Files are Fixed - Recreate Server Script:**

```bash
# Remove old server file
rm -f /workspace/llm_server.py

# Create new server with all fixes
cat > /workspace/llm_server.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn
import os

app = FastAPI()
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/Meta-Llama-3.1-8B-Instruct")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print(f"Loading official Meta Llama model from: {MODEL_PATH}...")
print("This may take 2-3 minutes on first load...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
print("Model loaded successfully!")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 250

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "quantization": "4-bit"
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(request.prompt):].strip()
    return {"generated_text": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
EOF

# Verify file was created
ls -lh /workspace/llm_server.py
cat /workspace/llm_server.py | wc -l
# Should show ~59 lines

echo "✅ Server script created successfully!"
```

---

## 🔧 **Useful Commands Inside RunPod**

### **Monitor GPU Usage**
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Press Ctrl+C to exit
```

### **Check Disk Space**
```bash
# Container disk (temporary)
df -h /

# Volume disk (persistent)
df -h /workspace
```

### **View Server Logs**
```bash
# If running in screen
screen -r llm-server

# If using systemd (see below)
journalctl -u llm-server -f
```

### **Check Running Processes**
```bash
# Check if server is running
ps aux | grep python

# Check ports
netstat -tulpn | grep 8888
```

---

## 🛡️ **Production Setup (Keep Server Running 24/7)**

### **Option 1: Screen (Simple)**
```bash
# Start
screen -S llm-server
python /workspace/llm_server.py

# Detach: Ctrl+A, D
# Reattach: screen -r llm-server
```

### **Option 2: Systemd Service (Recommended)**
```bash
# Create service file
cat > /etc/systemd/system/llm-server.service << 'EOF'
[Unit]
Description=LLM API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace
Environment="HF_TOKEN=hf_xxxxxxxxxxxxx"
Environment="MODEL_NAME=meta-llama/Llama-3.1-8B"
ExecStart=/usr/bin/python3 /workspace/llm_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

# Enable service (start on boot)
systemctl enable llm-server

# Start service
systemctl start llm-server

# Check status
systemctl status llm-server

# View logs
journalctl -u llm-server -f
```

---

## 📊 **Monitoring & Debugging**

### **Check Model Download Progress**
```bash
# Monitor model cache
du -sh /workspace/models/
watch -n 5 "du -sh /workspace/models/*"
```

### **Memory Usage**
```bash
# RAM usage
free -h

# GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### **Server Health**
```bash
# Check if FastAPI is responding
curl -v http://localhost:8888/health

# Check response time
time curl http://localhost:8888/health
```

---

## 🆘 **Troubleshooting**

### **Issue: "hf_transfer package is not available" during model download**
```bash
# Error: ValueError: Fast download using 'hf_transfer' is enabled but 'hf_transfer' package is not available

# Solution 1: Install hf_transfer (Recommended - much faster downloads)
pip install hf_transfer

# Then retry the download command
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False

# Solution 2: Disable fast transfer (slower but works without hf_transfer)
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0

# Then retry the download
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False
```

### **Issue: "Invalid user token" or "401 Unauthorized"**
```bash
# This means your HuggingFace token is invalid, expired, or has wrong permissions

# Step 1: Get a NEW token
# Go to: https://huggingface.co/settings/tokens
# Create new token with READ permissions

# Step 2: Clear old cached tokens
unset HF_TOKEN
rm -rf ~/.cache/huggingface/token

# Step 3: Login with NEW token
hf auth login --token hf_YOUR_NEW_TOKEN_HERE

# Step 4: Verify it works
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
# Should show your username and account info, NOT an error

# Step 5: Set environment variable for future use
export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE
echo "export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE" >> ~/.bashrc
```

### **Issue: "HF_TOKEN not found"**
```bash
# Set it manually
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Or login directly
hf auth login --token hf_xxxxxxxxxxxxxxxxxxxxx
```

### **Issue: "Out of memory"**
```bash
# Check GPU memory
nvidia-smi

# Reduce max_tokens in generation
# Edit llm_server.py and change default max_tokens from 250 to 150
```

### **Issue: "Model download is slow"**
```bash
# First download is slow (15GB model)
# Check progress:
ls -lh /workspace/models/

# Subsequent runs are instant (model is cached)
```

### **Issue: "Port 8888 already in use"**
```bash
# Check what's using it
lsof -i :8888

# Kill the process
kill -9 <PID>

# Or use different port
python llm_server.py --port 8889
```

### **Issue: "AttributeError: 'dict' object has no attribute 'model_type'" when starting server**

**This error occurs when model files are incomplete or corrupted during download.**

#### **Quick Fix - Check Model Files First:**

```bash
# Step 1: Check if config.json exists and is valid
cd /workspace/models/Meta-Llama-3.1-8B-Instruct
ls -lh config.json

# Step 2: Try to load the config with Python
python3 << 'EOF'
import json
try:
    with open('/workspace/models/Meta-Llama-3.1-8B-Instruct/config.json', 'r') as f:
        config = json.load(f)
    print("✅ Config file is valid JSON")
    print(f"Model type: {config.get('model_type', 'NOT FOUND')}")
    print(f"Architecture: {config.get('architectures', 'NOT FOUND')}")
except Exception as e:
    print(f"❌ Config file is corrupted: {e}")
EOF

# Step 3: Check if all essential model files exist
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct/ | grep -E "config.json|tokenizer|model.*safetensors"

# Expected files:
# - config.json (should be ~1-2KB)
# - tokenizer.json (should be ~1-2MB)
# - tokenizer_config.json
# - model-*.safetensors (should be multiple GB each)

# Step 4: Check total size and file count
du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct
ls /workspace/models/Meta-Llama-3.1-8B-Instruct | wc -l

# Expected: ~15-30GB total, at least 10-15 files
```

#### **If Config is Corrupted or Incomplete - Full Fix:**

```bash
# Step 1: Backup the corrupted model (if needed)
cd /workspace/models
mv Meta-Llama-3.1-8B-Instruct Meta-Llama-3.1-8B-Instruct.backup

# Step 2: Create fresh directory
mkdir -p /workspace/models/Meta-Llama-3.1-8B-Instruct

# Step 3: Verify you're logged into HuggingFace
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
# Should show your username, NOT an error

# Step 4: Install hf_transfer for reliable downloads (optional but recommended)
pip install hf_transfer

# Step 5: Re-download the model (this is the most reliable method)
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False \
  --resume-download

# The --resume-download flag will skip files that are already complete

# Step 6: Monitor download progress (in another terminal)
watch -n 5 "du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct && echo '---' && ls /workspace/models/Meta-Llama-3.1-8B-Instruct | wc -l"

# Step 7: After download completes, verify config.json
python3 << 'EOF'
import json
from transformers import AutoConfig

# Method 1: Check raw JSON
with open('/workspace/models/Meta-Llama-3.1-8B-Instruct/config.json', 'r') as f:
    config = json.load(f)
    print("✅ Config JSON is valid")
    print(f"   Model type: {config.get('model_type')}")
    print(f"   Architecture: {config.get('architectures')}")

# Method 2: Load with transformers (this is what the server uses)
try:
    config = AutoConfig.from_pretrained('/workspace/models/Meta-Llama-3.1-8B-Instruct', trust_remote_code=True)
    print("✅ Config loads correctly with transformers")
    print(f"   Model type: {config.model_type}")
except Exception as e:
    print(f"❌ Failed to load with transformers: {e}")
EOF

# Step 8: Verify all model shards are present
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct/*.safetensors

# Should see multiple files like:
# model-00001-of-00004.safetensors
# model-00002-of-00004.safetensors
# etc.

# Step 9: Once verified, recreate the server script (see next section)
```

#### **Alternative: If Download Keeps Failing - Download Individual Files:**

```bash
# Sometimes batch download fails, so download critical files individually

# Step 1: Create directory
mkdir -p /workspace/models/Meta-Llama-3.1-8B-Instruct
cd /workspace/models/Meta-Llama-3.1-8B-Instruct

# Step 2: Download config files first
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct generation_config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct tokenizer_config.json --local-dir . --local-dir-use-symlinks False

# Step 3: Download tokenizer files
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct tokenizer.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct special_tokens_map.json --local-dir . --local-dir-use-symlinks False

# Step 4: Verify config works before downloading large model files
python3 -c "from transformers import AutoConfig; config = AutoConfig.from_pretrained('.', trust_remote_code=True); print(f'✅ Config OK: {config.model_type}')"

# Step 5: Only if config works, download model weights (large files)
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --include "*.safetensors" --local-dir . --local-dir-use-symlinks False

# Step 6: Final verification
ls -lh
# Should see all files listed above
```

#### **After Model Files are Fixed - Recreate Server Script:**

```bash
# Remove old server file
rm -f /workspace/llm_server.py

# Create new server with all fixes
cat > /workspace/llm_server.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn
import os

app = FastAPI()
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/Meta-Llama-3.1-8B-Instruct")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print(f"Loading official Meta Llama model from: {MODEL_PATH}...")
print("This may take 2-3 minutes on first load...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
print("Model loaded successfully!")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 250

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "quantization": "4-bit"
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(request.prompt):].strip()
    return {"generated_text": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
EOF

# Verify file was created
ls -lh /workspace/llm_server.py
cat /workspace/llm_server.py | wc -l
# Should show ~59 lines

echo "✅ Server script created successfully!"
```

---

## 🔧 **Useful Commands Inside RunPod**

### **Monitor GPU Usage**
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Press Ctrl+C to exit
```

### **Check Disk Space**
```bash
# Container disk (temporary)
df -h /

# Volume disk (persistent)
df -h /workspace
```

### **View Server Logs**
```bash
# If running in screen
screen -r llm-server

# If using systemd (see below)
journalctl -u llm-server -f
```

### **Check Running Processes**
```bash
# Check if server is running
ps aux | grep python

# Check ports
netstat -tulpn | grep 8888
```

---

## 🛡️ **Production Setup (Keep Server Running 24/7)**

### **Option 1: Screen (Simple)**
```bash
# Start
screen -S llm-server
python /workspace/llm_server.py

# Detach: Ctrl+A, D
# Reattach: screen -r llm-server
```

### **Option 2: Systemd Service (Recommended)**
```bash
# Create service file
cat > /etc/systemd/system/llm-server.service << 'EOF'
[Unit]
Description=LLM API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace
Environment="HF_TOKEN=hf_xxxxxxxxxxxxx"
Environment="MODEL_NAME=meta-llama/Llama-3.1-8B"
ExecStart=/usr/bin/python3 /workspace/llm_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

# Enable service (start on boot)
systemctl enable llm-server

# Start service
systemctl start llm-server

# Check status
systemctl status llm-server

# View logs
journalctl -u llm-server -f
```

---

## 📊 **Monitoring & Debugging**

### **Check Model Download Progress**
```bash
# Monitor model cache
du -sh /workspace/models/
watch -n 5 "du -sh /workspace/models/*"
```

### **Memory Usage**
```bash
# RAM usage
free -h

# GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### **Server Health**
```bash
# Check if FastAPI is responding
curl -v http://localhost:8888/health

# Check response time
time curl http://localhost:8888/health
```

---

## 🆘 **Troubleshooting**

### **Issue: "hf_transfer package is not available" during model download**
```bash
# Error: ValueError: Fast download using 'hf_transfer' is enabled but 'hf_transfer' package is not available

# Solution 1: Install hf_transfer (Recommended - much faster downloads)
pip install hf_transfer

# Then retry the download command
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False

# Solution 2: Disable fast transfer (slower but works without hf_transfer)
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0

# Then retry the download
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False
```

### **Issue: "Invalid user token" or "401 Unauthorized"**
```bash
# This means your HuggingFace token is invalid, expired, or has wrong permissions

# Step 1: Get a NEW token
# Go to: https://huggingface.co/settings/tokens
# Create new token with READ permissions

# Step 2: Clear old cached tokens
unset HF_TOKEN
rm -rf ~/.cache/huggingface/token

# Step 3: Login with NEW token
hf auth login --token hf_YOUR_NEW_TOKEN_HERE

# Step 4: Verify it works
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
# Should show your username and account info, NOT an error

# Step 5: Set environment variable for future use
export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE
echo "export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE" >> ~/.bashrc
```

### **Issue: "HF_TOKEN not found"**
```bash
# Set it manually
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Or login directly
hf auth login --token hf_xxxxxxxxxxxxxxxxxxxxx
```

### **Issue: "Out of memory"**
```bash
# Check GPU memory
nvidia-smi

# Reduce max_tokens in generation
# Edit llm_server.py and change default max_tokens from 250 to 150
```

### **Issue: "Model download is slow"**
```bash
# First download is slow (15GB model)
# Check progress:
ls -lh /workspace/models/

# Subsequent runs are instant (model is cached)
```

### **Issue: "Port 8888 already in use"**
```bash
# Check what's using it
lsof -i :8888

# Kill the process
kill -9 <PID>

# Or use different port
python llm_server.py --port 8889
```

### **Issue: "AttributeError: 'dict' object has no attribute 'model_type'" when starting server**

**This error occurs when model files are incomplete or corrupted during download.**

#### **Quick Fix - Check Model Files First:**

```bash
# Step 1: Check if config.json exists and is valid
cd /workspace/models/Meta-Llama-3.1-8B-Instruct
ls -lh config.json

# Step 2: Try to load the config with Python
python3 << 'EOF'
import json
try:
    with open('/workspace/models/Meta-Llama-3.1-8B-Instruct/config.json', 'r') as f:
        config = json.load(f)
    print("✅ Config file is valid JSON")
    print(f"Model type: {config.get('model_type', 'NOT FOUND')}")
    print(f"Architecture: {config.get('architectures', 'NOT FOUND')}")
except Exception as e:
    print(f"❌ Config file is corrupted: {e}")
EOF

# Step 3: Check if all essential model files exist
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct/ | grep -E "config.json|tokenizer|model.*safetensors"

# Expected files:
# - config.json (should be ~1-2KB)
# - tokenizer.json (should be ~1-2MB)
# - tokenizer_config.json
# - model-*.safetensors (should be multiple GB each)

# Step 4: Check total size and file count
du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct
ls /workspace/models/Meta-Llama-3.1-8B-Instruct | wc -l

# Expected: ~15-30GB total, at least 10-15 files
```

#### **If Config is Corrupted or Incomplete - Full Fix:**

```bash
# Step 1: Backup the corrupted model (if needed)
cd /workspace/models
mv Meta-Llama-3.1-8B-Instruct Meta-Llama-3.1-8B-Instruct.backup

# Step 2: Create fresh directory
mkdir -p /workspace/models/Meta-Llama-3.1-8B-Instruct

# Step 3: Verify you're logged into HuggingFace
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
# Should show your username, NOT an error

# Step 4: Install hf_transfer for reliable downloads (optional but recommended)
pip install hf_transfer

# Step 5: Re-download the model (this is the most reliable method)
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False \
  --resume-download

# The --resume-download flag will skip files that are already complete

# Step 6: Monitor download progress (in another terminal)
watch -n 5 "du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct && echo '---' && ls /workspace/models/Meta-Llama-3.1-8B-Instruct | wc -l"

# Step 7: After download completes, verify config.json
python3 << 'EOF'
import json
from transformers import AutoConfig

# Method 1: Check raw JSON
with open('/workspace/models/Meta-Llama-3.1-8B-Instruct/config.json', 'r') as f:
    config = json.load(f)
    print("✅ Config JSON is valid")
    print(f"   Model type: {config.get('model_type')}")
    print(f"   Architecture: {config.get('architectures')}")

# Method 2: Load with transformers (this is what the server uses)
try:
    config = AutoConfig.from_pretrained('/workspace/models/Meta-Llama-3.1-8B-Instruct', trust_remote_code=True)
    print("✅ Config loads correctly with transformers")
    print(f"   Model type: {config.model_type}")
except Exception as e:
    print(f"❌ Failed to load with transformers: {e}")
EOF

# Step 8: Verify all model shards are present
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct/*.safetensors

# Should see multiple files like:
# model-00001-of-00004.safetensors
# model-00002-of-00004.safetensors
# etc.

# Step 9: Once verified, recreate the server script (see next section)
```

#### **Alternative: If Download Keeps Failing - Download Individual Files:**

```bash
# Sometimes batch download fails, so download critical files individually

# Step 1: Create directory
mkdir -p /workspace/models/Meta-Llama-3.1-8B-Instruct
cd /workspace/models/Meta-Llama-3.1-8B-Instruct

# Step 2: Download config files first
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct generation_config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct tokenizer_config.json --local-dir . --local-dir-use-symlinks False

# Step 3: Download tokenizer files
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct tokenizer.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct special_tokens_map.json --local-dir . --local-dir-use-symlinks False

# Step 4: Verify config works before downloading large model files
python3 -c "from transformers import AutoConfig; config = AutoConfig.from_pretrained('.', trust_remote_code=True); print(f'✅ Config OK: {config.model_type}')"

# Step 5: Only if config works, download model weights (large files)
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --include "*.safetensors" --local-dir . --local-dir-use-symlinks False

# Step 6: Final verification
ls -lh
# Should see all files listed above
```

#### **After Model Files are Fixed - Recreate Server Script:**

```bash
# Remove old server file
rm -f /workspace/llm_server.py

# Create new server with all fixes
cat > /workspace/llm_server.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn
import os

app = FastAPI()
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/Meta-Llama-3.1-8B-Instruct")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print(f"Loading official Meta Llama model from: {MODEL_PATH}...")
print("This may take 2-3 minutes on first load...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
print("Model loaded successfully!")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 250

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "quantization": "4-bit"
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(request.prompt):].strip()
    return {"generated_text": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
EOF

# Verify file was created
ls -lh /workspace/llm_server.py
cat /workspace/llm_server.py | wc -l
# Should show ~59 lines

echo "✅ Server script created successfully!"
```

---

## 🔧 **Useful Commands Inside RunPod**

### **Monitor GPU Usage**
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Press Ctrl+C to exit
```

### **Check Disk Space**
```bash
# Container disk (temporary)
df -h /

# Volume disk (persistent)
df -h /workspace
```

### **View Server Logs**
```bash
# If running in screen
screen -r llm-server

# If using systemd (see below)
journalctl -u llm-server -f
```

### **Check Running Processes**
```bash
# Check if server is running
ps aux | grep python

# Check ports
netstat -tulpn | grep 8888
```

---

## 🛡️ **Production Setup (Keep Server Running 24/7)**

### **Option 1: Screen (Simple)**
```bash
# Start
screen -S llm-server
python /workspace/llm_server.py

# Detach: Ctrl+A, D
# Reattach: screen -r llm-server
```

### **Option 2: Systemd Service (Recommended)**
```bash
# Create service file
cat > /etc/systemd/system/llm-server.service << 'EOF'
[Unit]
Description=LLM API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace
Environment="HF_TOKEN=hf_xxxxxxxxxxxxx"
Environment="MODEL_NAME=meta-llama/Llama-3.1-8B"
ExecStart=/usr/bin/python3 /workspace/llm_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

# Enable service (start on boot)
systemctl enable llm-server

# Start service
systemctl start llm-server

# Check status
systemctl status llm-server

# View logs
journalctl -u llm-server -f
```

---

## 📊 **Monitoring & Debugging**

### **Check Model Download Progress**
```bash
# Monitor model cache
du -sh /workspace/models/
watch -n 5 "du -sh /workspace/models/*"
```

### **Memory Usage**
```bash
# RAM usage
free -h

# GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### **Server Health**
```bash
# Check if FastAPI is responding
curl -v http://localhost:8888/health

# Check response time
time curl http://localhost:8888/health
```

---

## 🆘 **Troubleshooting**

### **Issue: "hf_transfer package is not available" during model download**
```bash
# Error: ValueError: Fast download using 'hf_transfer' is enabled but 'hf_transfer' package is not available

# Solution 1: Install hf_transfer (Recommended - much faster downloads)
pip install hf_transfer

# Then retry the download command
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False

# Solution 2: Disable fast transfer (slower but works without hf_transfer)
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0

# Then retry the download
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False
```

### **Issue: "Invalid user token" or "401 Unauthorized"**
```bash
# This means your HuggingFace token is invalid, expired, or has wrong permissions

# Step 1: Get a NEW token
# Go to: https://huggingface.co/settings/tokens
# Create new token with READ permissions

# Step 2: Clear old cached tokens
unset HF_TOKEN
rm -rf ~/.cache/huggingface/token

# Step 3: Login with NEW token
hf auth login --token hf_YOUR_NEW_TOKEN_HERE

# Step 4: Verify it works
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
# Should show your username and account info, NOT an error

# Step 5: Set environment variable for future use
export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE
echo "export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE" >> ~/.bashrc
```

### **Issue: "HF_TOKEN not found"**
```bash
# Set it manually
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Or login directly
hf auth login --token hf_xxxxxxxxxxxxxxxxxxxxx
```

### **Issue: "Out of memory"**
```bash
# Check GPU memory
nvidia-smi

# Reduce max_tokens in generation
# Edit llm_server.py and change default max_tokens from 250 to 150
```

### **Issue: "Model download is slow"**
```bash
# First download is slow (15GB model)
# Check progress:
ls -lh /workspace/models/

# Subsequent runs are instant (model is cached)
```

### **Issue: "Port 8888 already in use"**
```bash
# Check what's using it
lsof -i :8888

# Kill the process
kill -9 <PID>

# Or use different port
python llm_server.py --port 8889
```

### **Issue: "AttributeError: 'dict' object has no attribute 'model_type'" when starting server**

**This error occurs when model files are incomplete or corrupted during download.**

#### **Quick Fix - Check Model Files First:**

```bash
# Step 1: Check if config.json exists and is valid
cd /workspace/models/Meta-Llama-3.1-8B-Instruct
ls -lh config.json

# Step 2: Try to load the config with Python
python3 << 'EOF'
import json
try:
    with open('/workspace/models/Meta-Llama-3.1-8B-Instruct/config.json', 'r') as f:
        config = json.load(f)
    print("✅ Config file is valid JSON")
    print(f"Model type: {config.get('model_type', 'NOT FOUND')}")
    print(f"Architecture: {config.get('architectures', 'NOT FOUND')}")
except Exception as e:
    print(f"❌ Config file is corrupted: {e}")
EOF

# Step 3: Check if all essential model files exist
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct/ | grep -E "config.json|tokenizer|model.*safetensors"

# Expected files:
# - config.json (should be ~1-2KB)
# - tokenizer.json (should be ~1-2MB)
# - tokenizer_config.json
# - model-*.safetensors (should be multiple GB each)

# Step 4: Check total size and file count
du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct
ls /workspace/models/Meta-Llama-3.1-8B-Instruct | wc -l

# Expected: ~15-30GB total, at least 10-15 files
```

#### **If Config is Corrupted or Incomplete - Full Fix:**

```bash
# Step 1: Backup the corrupted model (if needed)
cd /workspace/models
mv Meta-Llama-3.1-8B-Instruct Meta-Llama-3.1-8B-Instruct.backup

# Step 2: Create fresh directory
mkdir -p /workspace/models/Meta-Llama-3.1-8B-Instruct

# Step 3: Verify you're logged into HuggingFace
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
# Should show your username, NOT an error

# Step 4: Install hf_transfer for reliable downloads (optional but recommended)
pip install hf_transfer

# Step 5: Re-download the model (this is the most reliable method)
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False \
  --resume-download

# The --resume-download flag will skip files that are already complete

# Step 6: Monitor download progress (in another terminal)
watch -n 5 "du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct && echo '---' && ls /workspace/models/Meta-Llama-3.1-8B-Instruct | wc -l"

# Step 7: After download completes, verify config.json
python3 << 'EOF'
import json
from transformers import AutoConfig

# Method 1: Check raw JSON
with open('/workspace/models/Meta-Llama-3.1-8B-Instruct/config.json', 'r') as f:
    config = json.load(f)
    print("✅ Config JSON is valid")
    print(f"   Model type: {config.get('model_type')}")
    print(f"   Architecture: {config.get('architectures')}")

# Method 2: Load with transformers (this is what the server uses)
try:
    config = AutoConfig.from_pretrained('/workspace/models/Meta-Llama-3.1-8B-Instruct', trust_remote_code=True)
    print("✅ Config loads correctly with transformers")
    print(f"   Model type: {config.model_type}")
except Exception as e:
    print(f"❌ Failed to load with transformers: {e}")
EOF

# Step 8: Verify all model shards are present
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct/*.safetensors

# Should see multiple files like:
# model-00001-of-00004.safetensors
# model-00002-of-00004.safetensors
# etc.

# Step 9: Once verified, recreate the server script (see next section)
```

#### **Alternative: If Download Keeps Failing - Download Individual Files:**

```bash
# Sometimes batch download fails, so download critical files individually

# Step 1: Create directory
mkdir -p /workspace/models/Meta-Llama-3.1-8B-Instruct
cd /workspace/models/Meta-Llama-3.1-8B-Instruct

# Step 2: Download config files first
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct generation_config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct tokenizer_config.json --local-dir . --local-dir-use-symlinks False

# Step 3: Download tokenizer files
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct tokenizer.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct special_tokens_map.json --local-dir . --local-dir-use-symlinks False

# Step 4: Verify config works before downloading large model files
python3 -c "from transformers import AutoConfig; config = AutoConfig.from_pretrained('.', trust_remote_code=True); print(f'✅ Config OK: {config.model_type}')"

# Step 5: Only if config works, download model weights (large files)
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --include "*.safetensors" --local-dir . --local-dir-use-symlinks False

# Step 6: Final verification
ls -lh
# Should see all files listed above
```

#### **After Model Files are Fixed - Recreate Server Script:**

```bash
# Remove old server file
rm -f /workspace/llm_server.py

# Create new server with all fixes
cat > /workspace/llm_server.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn
import os

app = FastAPI()
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/Meta-Llama-3.1-8B-Instruct")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print(f"Loading official Meta Llama model from: {MODEL_PATH}...")
print("This may take 2-3 minutes on first load...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
print("Model loaded successfully!")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 250

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "quantization": "4-bit"
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(request.prompt):].strip()
    return {"generated_text": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
EOF

# Verify file was created
ls -lh /workspace/llm_server.py
cat /workspace/llm_server.py | wc -l
# Should show ~59 lines

echo "✅ Server script created successfully!"
```

---

## 🔧 **Useful Commands Inside RunPod**

### **Monitor GPU Usage**
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Press Ctrl+C to exit
```

### **Check Disk Space**
```bash
# Container disk (temporary)
df -h /

# Volume disk (persistent)
df -h /workspace
```

### **View Server Logs**
```bash
# If running in screen
screen -r llm-server

# If using systemd (see below)
journalctl -u llm-server -f
```

### **Check Running Processes**
```bash
# Check if server is running
ps aux | grep python

# Check ports
netstat -tulpn | grep 8888
```

---

## 🛡️ **Production Setup (Keep Server Running 24/7)**

### **Option 1: Screen (Simple)**
```bash
# Start
screen -S llm-server
python /workspace/llm_server.py

# Detach: Ctrl+A, D
# Reattach: screen -r llm-server
```

### **Option 2: Systemd Service (Recommended)**
```bash
# Create service file
cat > /etc/systemd/system/llm-server.service << 'EOF'
[Unit]
Description=LLM API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace
Environment="HF_TOKEN=hf_xxxxxxxxxxxxx"
Environment="MODEL_NAME=meta-llama/Llama-3.1-8B"
ExecStart=/usr/bin/python3 /workspace/llm_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

# Enable service (start on boot)
systemctl enable llm-server

# Start service
systemctl start llm-server

# Check status
systemctl status llm-server

# View logs
journalctl -u llm-server -f
```

---

## 📊 **Monitoring & Debugging**

### **Check Model Download Progress**
```bash
# Monitor model cache
du -sh /workspace/models/
watch -n 5 "du -sh /workspace/models/*"
```

### **Memory Usage**
```bash
# RAM usage
free -h

# GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### **Server Health**
```bash
# Check if FastAPI is responding
curl -v http://localhost:8888/health

# Check response time
time curl http://localhost:8888/health
```

---

## 🆘 **Troubleshooting**

### **Issue: "hf_transfer package is not available" during model download**
```bash
# Error: ValueError: Fast download using 'hf_transfer' is enabled but 'hf_transfer' package is not available

# Solution 1: Install hf_transfer (Recommended - much faster downloads)
pip install hf_transfer

# Then retry the download command
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False

# Solution 2: Disable fast transfer (slower but works without hf_transfer)
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0

# Then retry the download
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False
```

### **Issue: "Invalid user token" or "401 Unauthorized"**
```bash
# This means your HuggingFace token is invalid, expired, or has wrong permissions

# Step 1: Get a NEW token
# Go to: https://huggingface.co/settings/tokens
# Create new token with READ permissions

# Step 2: Clear old cached tokens
unset HF_TOKEN
rm -rf ~/.cache/huggingface/token

# Step 3: Login with NEW token
hf auth login --token hf_YOUR_NEW_TOKEN_HERE

# Step 4: Verify it works
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
# Should show your username and account info, NOT an error

# Step 5: Set environment variable for future use
export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE
echo "export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE" >> ~/.bashrc
```

### **Issue: "HF_TOKEN not found"**
```bash
# Set it manually
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Or login directly
hf auth login --token hf_xxxxxxxxxxxxxxxxxxxxx
```

### **Issue: "Out of memory"**
```bash
# Check GPU memory
nvidia-smi

# Reduce max_tokens in generation
# Edit llm_server.py and change default max_tokens from 250 to 150
```

### **Issue: "Model download is slow"**
```bash
# First download is slow (15GB model)
# Check progress:
ls -lh /workspace/models/

# Subsequent runs are instant (model is cached)
```

### **Issue: "Port 8888 already in use"**
```bash
# Check what's using it
lsof -i :8888

# Kill the process
kill -9 <PID>

# Or use different port
python llm_server.py --port 8889
```

### **Issue: "AttributeError: 'dict' object has no attribute 'model_type'" when starting server**

**This error occurs when model files are incomplete or corrupted during download.**

#### **Quick Fix - Check Model Files First:**

```bash
# Step 1: Check if config.json exists and is valid
cd /workspace/models/Meta-Llama-3.1-8B-Instruct
ls -lh config.json

# Step 2: Try to load the config with Python
python3 << 'EOF'
import json
try:
    with open('/workspace/models/Meta-Llama-3.1-8B-Instruct/config.json', 'r') as f:
        config = json.load(f)
    print("✅ Config file is valid JSON")
    print(f"Model type: {config.get('model_type', 'NOT FOUND')}")
    print(f"Architecture: {config.get('architectures', 'NOT FOUND')}")
except Exception as e:
    print(f"❌ Config file is corrupted: {e}")
EOF

# Step 3: Check if all essential model files exist
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct/ | grep -E "config.json|tokenizer|model.*safetensors"

# Expected files:
# - config.json (should be ~1-2KB)
# - tokenizer.json (should be ~1-2MB)
# - tokenizer_config.json
# - model-*.safetensors (should be multiple GB each)

# Step 4: Check total size and file count
du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct
ls /workspace/models/Meta-Llama-3.1-8B-Instruct | wc -l

# Expected: ~15-30GB total, at least 10-15 files
```

#### **If Config is Corrupted or Incomplete - Full Fix:**

```bash
# Step 1: Backup the corrupted model (if needed)
cd /workspace/models
mv Meta-Llama-3.1-8B-Instruct Meta-Llama-3.1-8B-Instruct.backup

# Step 2: Create fresh directory
mkdir -p /workspace/models/Meta-Llama-3.1-8B-Instruct

# Step 3: Verify you're logged into HuggingFace
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
# Should show your username, NOT an error

# Step 4: Install hf_transfer for reliable downloads (optional but recommended)
pip install hf_transfer

# Step 5: Re-download the model (this is the most reliable method)
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False \
  --resume-download

# The --resume-download flag will skip files that are already complete

# Step 6: Monitor download progress (in another terminal)
watch -n 5 "du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct && echo '---' && ls /workspace/models/Meta-Llama-3.1-8B-Instruct | wc -l"

# Step 7: After download completes, verify config.json
python3 << 'EOF'
import json
from transformers import AutoConfig

# Method 1: Check raw JSON
with open('/workspace/models/Meta-Llama-3.1-8B-Instruct/config.json', 'r') as f:
    config = json.load(f)
    print("✅ Config JSON is valid")
    print(f"   Model type: {config.get('model_type')}")
    print(f"   Architecture: {config.get('architectures')}")

# Method 2: Load with transformers (this is what the server uses)
try:
    config = AutoConfig.from_pretrained('/workspace/models/Meta-Llama-3.1-8B-Instruct', trust_remote_code=True)
    print("✅ Config loads correctly with transformers")
    print(f"   Model type: {config.model_type}")
except Exception as e:
    print(f"❌ Failed to load with transformers: {e}")
EOF

# Step 8: Verify all model shards are present
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct/*.safetensors

# Should see multiple files like:
# model-00001-of-00004.safetensors
# model-00002-of-00004.safetensors
# etc.

# Step 9: Once verified, recreate the server script (see next section)
```

#### **Alternative: If Download Keeps Failing - Download Individual Files:**

```bash
# Sometimes batch download fails, so download critical files individually

# Step 1: Create directory
mkdir -p /workspace/models/Meta-Llama-3.1-8B-Instruct
cd /workspace/models/Meta-Llama-3.1-8B-Instruct

# Step 2: Download config files first
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct generation_config.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct tokenizer_config.json --local-dir . --local-dir-use-symlinks False

# Step 3: Download tokenizer files
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct tokenizer.json --local-dir . --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct special_tokens_map.json --local-dir . --local-dir-use-symlinks False

# Step 4: Verify config works before downloading large model files
python3 -c "from transformers import AutoConfig; config = AutoConfig.from_pretrained('.', trust_remote_code=True); print(f'✅ Config OK: {config.model_type}')"

# Step 5: Only if config works, download model weights (large files)
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --include "*.safetensors" --local-dir . --local-dir-use-symlinks False

# Step 6: Final verification
ls -lh
# Should see all files listed above
```

#### **After Model Files are Fixed - Recreate Server Script:**

```bash
# Remove old server file
rm -f /workspace/llm_server.py

# Create new server with all fixes
cat > /workspace/llm_server.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn
import os

app = FastAPI()
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/Meta-Llama-3.1-8B-Instruct")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print(f"Loading official Meta Llama model from: {MODEL_PATH}...")
print("This may take 2-3 minutes on first load...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
print("Model loaded successfully!")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 250

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "quantization": "4-bit"
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(request.prompt):].strip()
    return {"generated_text": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
EOF

# Verify file was created
ls -lh /workspace/llm_server.py
cat /workspace/llm_server.py | wc -l
# Should show ~59 lines

echo "✅ Server script created successfully!"
```

---

## 🔧 **Useful Commands Inside RunPod**

### **Monitor GPU Usage**
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Press Ctrl+C to exit
```

### **Check Disk Space**
```bash
# Container disk (temporary)
df -h /

# Volume disk (persistent)
df -h /workspace
```

### **View Server Logs**
```bash
# If running in screen
screen -r llm-server

# If using systemd (see below)
journalctl -u llm-server -f
```

### **Check Running Processes**
```bash
# Check if server is running
ps aux | grep python

# Check ports
netstat -tulpn | grep 8888
```

---

## 🛡️ **Production Setup (Keep Server Running 24/7)**

### **Option 1: Screen (Simple)**
```bash
# Start
screen -S llm-server
python /workspace/llm_server.py

# Detach: Ctrl+A, D
# Reattach: screen -r llm-server
```

### **Option 2: Systemd Service (Recommended)**
```bash
# Create service file
cat > /etc/systemd/system/llm-server.service << 'EOF'
[Unit]
Description=LLM API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace
Environment="HF_TOKEN=hf_xxxxxxxxxxxxx"
Environment="MODEL_NAME=meta-llama/Llama-3.1-8B"
ExecStart=/usr/bin/python3 /workspace/llm_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

# Enable service (start on boot)
systemctl enable llm-server

# Start service
systemctl start llm-server

# Check status
systemctl status llm-server

# View logs
journalctl -u llm-server -f
```

---

## 📊 **Monitoring & Debugging**

### **Check Model Download Progress**
```bash
# Monitor model cache
du -sh /workspace/models/
watch -n 5 "du -sh /workspace/models/*"
```

### **Memory Usage**
```bash
# RAM usage
free -h

# GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### **Server Health**
```bash
# Check if FastAPI is responding
curl -v http://localhost:8888/health

# Check response time
time curl http://localhost:8888/health
```

---

## 🆘 **Troubleshooting**

### **Issue: "hf_transfer package is not available" during model download**
```bash
# Error: ValueError: Fast download using 'hf_transfer' is enabled but 'hf_transfer' package is not available

# Solution 1: Install hf_transfer (Recommended - much faster downloads)
pip install hf_transfer

# Then retry the download command
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False

# Solution 2: Disable fast transfer (slower but works without hf_transfer)
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0

# Then retry the download
huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False
```

### **Issue: "Invalid user token" or "401 Unauthorized"**
```bash
# This means your HuggingFace token is invalid, expired, or has wrong permissions

# Step 1: Get a NEW token
# Go to: https://huggingface.co/settings/tokens
# Create new token with READ permissions

# Step 2: Clear old cached tokens
unset HF_TOKEN
rm -rf ~/.cache/huggingface/token

# Step 3: Login with NEW token
hf auth login --token hf_YOUR_NEW_TOKEN_HERE

# Step 4: Verify it works
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
# Should show your username and account info, NOT an error

# Step 5: Set environment variable for future use
export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE
echo "export HF_TOKEN=hf_YOUR_NEW_TOKEN_HERE" >> ~/.bashrc
```

### **Issue: "HF_TOKEN not found"**
```bash
# Set it manually
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Or login directly
hf auth login --token hf_xxxxxxxxxxxxxxxxxxxxx
```

### **Issue: "Out of memory"**
```bash
# Check GPU memory
nvidia-smi

# Reduce max_tokens in generation
# Edit llm_server.py and change default max_tokens from 250 to 150
```

### **Issue: "Model download is slow"**
```bash
# First download is slow (15GB model)
# Check progress:
ls -lh /workspace/models/

# Subsequent runs are instant (model is cached)
```

### **Issue: "Port 8888 already in use"**
```bash
# Check what's using it
lsof -i :8888

# Kill the process
kill -9 <PID>

# Or use different port
python llm_server.py --port 8889
```

### **Issue: "AttributeError: 'dict' object has no attribute 'model_type'" when starting server**

**This error occurs when model files are incomplete or corrupted during download.**

#### **Quick Fix - Check Model Files First:**

```bash
# Step 1: Check if config.json exists and is valid
cd /workspace/models/Meta-Llama-3.1-8B-Instruct
ls -lh config.json

# Step 2: Try to load the config with Python
python3 << 'EOF'
import json
try:
    with open('/workspace/models/Meta-Llama-3.1-8B-Instruct/config.json', 'r') as f:
        config = json.load(f)
    print("✅ Config file is valid JSON")
    print(f"Model type: {config.get('model_type', 'NOT FOUND')}")
    print(f"Architecture: {config.get('architectures', 'NOT FOUND')}")
except Exception as e: