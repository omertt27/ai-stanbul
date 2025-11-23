# 🚀 RunPod LLM Server - 5 Minute Setup

## 🔑 **BEFORE YOU START - GET YOUR TOKEN!**

1. **Go to:** https://huggingface.co/settings/tokens
2. **Click:** "New token" (or copy existing one)
3. **Name:** `runpod-llama`
4. **Copy the token** - it looks like: `hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890`
5. **SAVE IT** - you'll need it in Step 3 below!

> **⚠️ Common Mistake:** Don't literally type `YOUR_HF_TOKEN_HERE` - use your ACTUAL token!

---

## ⚡ Quick Start (2 Methods)

### 🎯 **METHOD 1: One-Command Setup (FASTEST)**

1. **Open RunPod Web Terminal** at your pod dashboard

2. **Copy-paste these commands in ONE GO:**
```bash
# Download setup script and LLM server
cd ~ && \
wget https://raw.githubusercontent.com/YOUR_GITHUB/ai-stanbul/main/setup_runpod_llm.sh && \
wget https://raw.githubusercontent.com/YOUR_GITHUB/ai-stanbul/main/llm_api_server_4bit.py && \
chmod +x setup_runpod_llm.sh && \
bash setup_runpod_llm.sh && \
python3 llm_api_server_4bit.py
```

---

### 📤 **METHOD 2: Copy-Paste Setup (RECOMMENDED)**

#### **STEP 1: Get Hugging Face Token** 🔑

**IMPORTANT:** Llama 3.1 8B requires authentication!

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"** or copy existing token
3. Give it a name: `runpod-llama`
4. **Copy the token** (starts with `hf_...`)

**Keep this token ready for Step 2!**

#### **STEP 2: Open RunPod Terminal**

1. Go to your pod dashboard
2. Click **"Connect"** → **"Start Web Terminal"**
3. You should see a terminal window

#### **STEP 3: Login to Hugging Face**

**⚠️ IMPORTANT: Replace `hf_YOUR_ACTUAL_TOKEN` with your real token!**

```bash
# Install huggingface-hub (if not already installed)
pip install huggingface-hub
```

**Then login with YOUR actual token:**
```bash
huggingface-cli login --token hf_YOUR_ACTUAL_TOKEN
```

**Real Example (use YOUR token, not this one):**
```bash
# This is just an example - use your own token from https://huggingface.co/settings/tokens
huggingface-cli login --token hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890
```

**✅ Expected output (ignore the deprecation warning):**
```
⚠️  Warning: 'huggingface-cli login' is deprecated. Use 'hf auth login' instead.
Token is valid (permission: read).
Your token has been saved to /root/.cache/huggingface/token
Login successful
```

**❌ If you see "Invalid user token":**
- You didn't replace `YOUR_HF_TOKEN_HERE` with your actual token
- Or your token is incorrect/expired
- Go get a new token from: https://huggingface.co/settings/tokens

#### **STEP 4: Install Dependencies**
#### **STEP 4: Install Dependencies**

**Copy-paste these commands ONE BY ONE:**

```bash
# Update system
apt-get update -y && apt-get upgrade -y
```

```bash
# Install Python and tools
apt-get install -y python3-pip git wget curl nano
```

```bash
# Install PyTorch with CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```bash
# Install ML libraries
pip3 install transformers accelerate bitsandbytes scipy
```

```bash
# Install web server libraries
pip3 install fastapi uvicorn pydantic
```

```bash
# Verify GPU works
nvidia-smi
```

**Expected:** You should see your GPU listed (RTX 3090, A100, etc.)

#### **STEP 5: Create LLM Server File**

**Copy-paste this ENTIRE block into the terminal:**

```bash
cat > llm_api_server_4bit.py << 'ENDOFFILE'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import uvicorn
import os

app = FastAPI(title="AI Istanbul LLM API (4-bit)")

# Global variables
model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 256
    temperature: float = 0.7

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print("🚀 Loading Llama 3.1 8B with 4-bit quantization...")
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Check HF token
    if not os.getenv("HF_TOKEN") and not os.path.exists("/root/.cache/huggingface/token"):
        print("⚠️  WARNING: No Hugging Face token found!")
        print("   Run: huggingface-cli login --token YOUR_TOKEN")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=True  # Uses saved HF token
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded successfully!")
    print(f"🔥 GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "gpu_memory_gb": f"{torch.cuda.memory_allocated()/1024**3:.1f}" if torch.cuda.is_available() else "N/A"
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Istanbul-focused system prompt
    system_prompt = """You are AI Istanbul, an expert travel assistant for Istanbul, Turkey.
Provide helpful, accurate, and concise information about Istanbul's attractions, restaurants, 
culture, transportation, and hidden gems. Include coordinates when mentioning locations."""
    
    # Llama 3.1 chat template format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.prompt}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_length,
        temperature=request.temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    
    return {
        "response": response,
        "model": "meta-llama/Llama-3.1-8B-Instruct"
    }

@app.post("/istanbul-query")
async def istanbul_query(request: GenerateRequest):
    """Specialized endpoint for Istanbul queries"""
    return await generate(request)

if __name__ == "__main__":
    print("🚀 Starting AI Istanbul LLM API Server (Llama 3.1 8B 4-bit)")
    print("📍 Port: 8888")
    print("🔐 Using Hugging Face token from cache")
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")
ENDOFFILE
```

**Press ENTER** - file created!

#### **STEP 6: Start the LLM Server**
#### **STEP 6: Start the LLM Server**

```bash
python3 llm_api_server_4bit.py
```

**This will:**
1. Download Llama 3.1 8B model (~8GB) - takes 3-7 minutes
2. Load it with 4-bit quantization (uses ~5-6 GB GPU RAM)
3. Start API server on port 8888

**Expected Output:**
```
🚀 Starting AI Istanbul LLM API Server (Llama 3.1 8B 4-bit)
� Port: 8888
🔐 Using Hugging Face token from cache
INFO:     Started server process
🚀 Loading Llama 3.1 8B with 4-bit quantization...
Downloading model files...
Loading checkpoint shards: 100%|██████████| 4/4
✅ Model loaded successfully!
🔥 GPU Memory: 5.8 GB
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8888
```

**🎉 If you see this, the LLM server is LIVE!**

---

## ✅ **Verify It's Working**

### **Test 1: Health Check**
```bash
# In a new terminal tab
curl http://localhost:8888/health
```
**Expected:** `{"status":"healthy","model_loaded":true,"model":"meta-llama/Llama-3.1-8B-Instruct"}`

### **Test 2: Generate Response**
```bash
curl -X POST http://localhost:8888/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Tell me about Hagia Sophia","max_length":100}'
```

### **Test 3: From Your Local Machine**
```bash
# Replace with YOUR proxy URL
curl https://4vq1b984pitw8s-8888.proxy.runpod.net/health
```

---

## 🔧 **Troubleshooting**

### **If model download is slow:**
```bash
# Pre-download model first (optional)
python3 -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')"
```

### **If GPU out of memory:**
Edit `llm_api_server_4bit.py`, line 95:
```python
max_new_tokens=128  # Reduce from 256
```

### **If port 8888 is busy:**
Edit `llm_api_server_4bit.py`, line 358:
```python
uvicorn.run(app, host="0.0.0.0", port=9999)  # Change to 9999
```

---

## 🎯 **Next Steps**

Once the server is running:

1. **Test the proxy URL:**
   ```bash
   curl https://4vq1b984pitw8s-8888.proxy.runpod.net/health
   ```

2. **Update Backend .env:**
   ```bash
   LLM_API_URL=https://4vq1b984pitw8s-8888.proxy.runpod.net
   ML_SERVICE_ENABLED=false
   ```

3. **Restart Backend:**
   ```bash
   # On Render dashboard
   Manual Deploy → Deploy latest commit
   ```

4. **Test End-to-End:**
   ```bash
   curl -X POST https://api.yourdomain.com/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"message":"What are the best restaurants in Sultanahmet?","session_id":"test123"}'
   ```

---

## 📋 **Quick Reference**

| Command | Purpose |
|---------|---------|
| `python3 llm_api_server_4bit.py` | Start LLM server |
| `curl localhost:8888/health` | Check server health |
| `nvidia-smi` | Check GPU usage |
| `ps aux \| grep python` | Find running Python processes |
| `kill -9 <PID>` | Stop LLM server |

---

## 🆘 **Need Help?**

**Server won't start?**
- Check logs: `tail -f ~/llm_server.log`
- Check GPU: `nvidia-smi`
- Check disk space: `df -h`

**Still stuck?**
Check these files for more details:
- `RUNPOD_SETUP_INSTRUCTIONS.md` (detailed guide)
- `RUNPOD_QUICK_START.md` (step-by-step)
- `FINAL_15_MIN_CHECKLIST.md` (full deployment)

---

## 🎉 **Success Checklist**

- [ ] RunPod pod is running
- [ ] Files uploaded (llm_api_server_4bit.py + setup_runpod_llm.sh)
- [ ] Setup script completed successfully
- [ ] LLM server is running on port 8888
- [ ] Health check returns "healthy"
- [ ] Proxy URL is accessible from internet
- [ ] Backend .env updated with LLM_API_URL
- [ ] Backend restarted on Render
- [ ] Chat endpoint returns real AI responses (not templates)

**Once all checked, you're LIVE! 🚀**
