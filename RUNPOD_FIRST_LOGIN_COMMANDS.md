# RunPod First Login Commands

**Connect to your RunPod pod first!** 🎉

---

## 🔐 **Step 0: SSH into Your Pod**

**⚠️ IMPORTANT: Run on YOUR LOCAL TERMINAL (Mac), not inside RunPod!**

**Option A: SSH via RunPod Proxy (Recommended)**
```bash
ssh vn290bqt32835t-64410fd1@ssh.runpod.io -i ~/.ssh/id_ed25519
```

**Option B: Direct TCP Connection (Supports SCP & SFTP)**
```bash
ssh root@194.68.245.166 -p 22124 -i ~/.ssh/id_ed25519
```

**Option C: If SSH key not found, try without `-i` flag:**
```bash
ssh vn290bqt32835t-64410fd1@ssh.runpod.io
```

Use Option A for general terminal access. Use Option B if you need to transfer files.

Once connected, your prompt looks like: `root@d39ec100f552:/#`

---

## 🚀 **Quick Setup (Copy & Paste These Commands)**

### **Step 1: Update System & Install Dependencies**

```bash
# Update pip
pip install --upgrade pip

# Install required packages for LLM server (including huggingface_hub for CLI)
pip install transformers accelerate bitsandbytes fastapi uvicorn huggingface_hub

# Verify installations
pip list | grep -E "transformers|accelerate|bitsandbytes|fastapi|uvicorn|huggingface"
```

---

### **Step 2: Login to HuggingFace**

```bash
# Login with your HuggingFace token
huggingface-cli login --token $HF_TOKEN

# If HF_TOKEN environment variable is not set, use this instead:
# huggingface-cli login --token hf_xxxxxxxxxxxxxxxxxxxxx

# Verify login
huggingface-cli whoami

# Expected output: Your HuggingFace username
```

**⚠️ If you get "command not found":**
```bash
# Install huggingface_hub package
pip install huggingface_hub

# Or use Python directly to login:
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

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

```bash
# Navigate to persistent storage
cd /workspace

# Create the server script
cat > llm_server.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn
import os

app = FastAPI()

# Get model name from environment variable or use default
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B")

# If full URL provided, extract model name
if MODEL_NAME.startswith("https://huggingface.co/"):
    MODEL_NAME = MODEL_NAME.replace("https://huggingface.co/", "")

# Load model in 4-bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

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

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 250

@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_NAME}

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
EOF

# Verify the file was created
ls -lh llm_server.py
cat llm_server.py | head -20
```

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
cd /workspace
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

### **Issue: "HF_TOKEN not found"**
```bash
# Set it manually
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Or login directly
huggingface-cli login --token hf_xxxxxxxxxxxxxxxxxxxxx
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

---

## ✅ **Success Checklist**

- [ ] Connected to RunPod pod via SSH or Web Terminal
- [ ] Installed dependencies (transformers, fastapi, etc.)
- [ ] Logged into HuggingFace (`huggingface-cli whoami` works)
- [ ] GPU detected (`nvidia-smi` shows GPU)
- [ ] Created `llm_server.py` in `/workspace`
- [ ] Started server (screen or systemd)
- [ ] Local health check works (`curl http://localhost:8888/health`)
- [ ] Got public endpoint from RunPod Console
- [ ] Public endpoint works (`curl https://<pod-id>-8888.proxy.runpod.net/health`)
- [ ] Test generation works
- [ ] Server running in background (screen or systemd)

---

## 🎯 **Next Steps**

1. **Copy your public endpoint URL:**
   ```
   https://<your-pod-id>-8888.proxy.runpod.net
   ```

2. **Update your backend `.env` file:**
   ```bash
   LLM_API_URL=https://<your-pod-id>-8888.proxy.runpod.net
   ```

3. **Test from your backend:**
   ```bash
   curl -X POST $LLM_API_URL/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Test from backend", "max_tokens": 50}'
   ```

4. **Integrate with frontend** (no changes needed - goes through backend)

---

## 📚 **Useful RunPod Commands**

```bash
# Exit pod (doesn't stop it)
exit

# Reconnect to pod
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519

# Or use Web Terminal in RunPod Console
```

---

**🎉 Your LLM server is now running on RunPod!**

**Your Public Endpoint:** `https://<your-pod-id>-8888.proxy.runpod.net`

**Test it from anywhere:**
```bash
curl https://<your-pod-id>-8888.proxy.runpod.net/health
```

For more help, see: [RUNPOD_DEPLOYMENT_GUIDE.md](./RUNPOD_DEPLOYMENT_GUIDE.md)
