# 🚀 VM Setup - Remaining Steps
**Continuing from Browser SSH Installation**

**Status:** Docker, Python, and Git installed via Browser SSH  
**Next:** Clone repository, transfer model, and launch API

---

## ✅ STEP 5: Reconnect from Your Mac Terminal

Close the browser SSH window and run this in your Mac terminal:

```bash
# Reconnect with gcloud
gcloud compute ssh instance-20251109-085407 --zone=europe-west1-b

# You should see the VM prompt:
# omer@instance-20251109-085407:~$
```

**Test that Docker works (without sudo):**

```bash
# Check docker version
docker --version

# Expected output: Docker version 24.x.x, build...

# Test docker
docker ps

# Expected output: 
# CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
# (empty list is fine - no containers running yet)
```

**If docker commands work without sudo, you're good to go! ✅**

---

## 📦 STEP 6: Clone Repository (2 minutes)

```bash
# Clone your AI Istanbul repository
git clone https://github.com/YOUR_USERNAME/ai-stanbul.git

# If you don't have it on GitHub, we'll create the files manually
# For now, let's create the project directory:
mkdir -p ~/ai-stanbul
cd ~/ai-stanbul

# Create models directory
mkdir -p models
```

---

## 📥 STEP 7: Transfer Llama Model from Your Mac (20-30 minutes)

**On your Mac (open a NEW terminal window, don't close the SSH connection):**

```bash
# First, check if you have the model locally
cd /Users/omer/Desktop/ai-stanbul
ls -lh models/

# If you see a llama-3.1-8b directory, proceed with transfer:
gcloud compute scp --recurse models/llama-3.1-8b \
  instance-20251109-085407:~/ai-stanbul/models/ \
  --zone=europe-west1-b

# This will take 20-30 minutes (model is ~16GB)
# You'll see progress like:
# config.json                   100%  743   1.5KB/s   00:00
# model-00001-of-00004.safetensors  25%  4.2GB  140MB/s   00:10
```

**While that's transferring, let's prepare the API server files...**

---

## 📝 STEP 8: Create API Server Files (5 minutes)

**Back in your SSH terminal (keep the model transfer running in the other terminal):**

```bash
# Make sure you're in the right directory
cd ~/ai-stanbul

# Create requirements.txt
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
transformers==4.35.0
torch==2.1.0
accelerate==0.24.1
bitsandbytes==0.41.1
pydantic==2.5.0
requests==2.31.0
python-dotenv==1.0.0
EOF

# Create the LLM API server
cat > llm_api_server.py << 'EOF'
"""
LLM Inference API Server for Google Cloud VM
Serves Llama 3.1 8B model via FastAPI
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Istanbul LLM API", version="1.0")

# Global model variables
model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7
    
class GenerateResponse(BaseModel):
    response: str
    tokens_generated: int
    model_info: dict = None

@app.on_event("startup")
async def load_model():
    """Load Llama 3.1 8B model on startup"""
    global model, tokenizer
    
    logger.info("🚀 Starting AI Istanbul LLM API...")
    logger.info("📂 Loading Llama 3.1 8B model...")
    
    model_path = os.getenv("MODEL_PATH", "./models/llama-3.1-8b")
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found at: {model_path}")
        logger.info("💡 Please transfer the model to the VM first")
        return
    
    try:
        # Load tokenizer
        logger.info("📖 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model (CPU, FP32)
        logger.info("🧠 Loading model (this takes 2-3 minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # Set to evaluation mode
        model.eval()
        
        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        
        logger.info("✅ Model loaded successfully!")
        logger.info(f"📊 Model parameters: {num_params/1e9:.2f}B")
        logger.info(f"💾 Device: CPU")
        logger.info(f"🔧 Dtype: {model.dtype}")
        logger.info("🎯 Ready to serve requests!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error(f"💡 Ensure model files exist at: {model_path}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AI Istanbul LLM API",
        "version": "1.0",
        "model": "Llama 3.1 8B",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "chat": "/chat"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": "cpu",
        "ready": model is not None and tokenizer is not None
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text using Llama 3.1 8B"""
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please wait for model to load or check logs."
        )
    
    try:
        logger.info(f"🔄 Generating response for prompt: {request.prompt[:100]}...")
        
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt")
        input_length = len(inputs.input_ids[0])
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        response_text = generated_text[len(request.prompt):].strip()
        
        tokens_generated = len(outputs[0]) - input_length
        
        logger.info(f"✅ Generated {tokens_generated} tokens, {len(response_text)} characters")
        
        return GenerateResponse(
            response=response_text,
            tokens_generated=tokens_generated,
            model_info={
                "model": "Llama-3.1-8B",
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/chat")
async def chat(request: GenerateRequest):
    """Chat endpoint with conversation formatting"""
    
    # Format as chat message
    formatted_prompt = f"""You are a helpful AI assistant for Istanbul tourism.
Provide accurate, friendly, and concise information about Istanbul.

User: {request.prompt}
Assistant:"""
    
    # Call generate with formatted prompt
    result = await generate(GenerateRequest(
        prompt=formatted_prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    ))
    
    return result

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"🚀 Starting LLM API server on 0.0.0.0:{port}")
    logger.info(f"📍 Access at: http://localhost:{port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
EOF

echo "✅ API server files created!"
```

---

## 🔥 STEP 9: Wait for Model Transfer to Complete

**Check the transfer progress in your Mac terminal:**

```bash
# You should see something like:
# Uploading models/llama-3.1-8b/model-00001-of-00004.safetensors
# [=====>                                    ]  25% (4.2GB/16GB)
```

**Once transfer is complete (all files uploaded), verify on VM:**

```bash
# In your SSH terminal:
ls -lh ~/ai-stanbul/models/llama-3.1-8b/

# You should see:
# config.json
# generation_config.json
# model-00001-of-00004.safetensors
# model-00002-of-00004.safetensors
# model-00003-of-00004.safetensors
# model-00004-of-00004.safetensors
# tokenizer.json
# tokenizer_config.json
# ... (and other files)
```

---

## 🚀 STEP 10: Install Python Dependencies (5 minutes)

```bash
# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# This will take 3-5 minutes to download and install all packages
```

---

## 🎯 STEP 11: Test API Server Locally (10 minutes)

```bash
# Start the API server
python llm_api_server.py

# You should see:
# 🚀 Starting AI Istanbul LLM API...
# 📂 Loading Llama 3.1 8B model...
# 📖 Loading tokenizer...
# 🧠 Loading model (this takes 2-3 minutes)...
# ✅ Model loaded successfully!
# 📊 Model parameters: 8.03B
# 💾 Device: CPU
# 🎯 Ready to serve requests!
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Keep this terminal running!**

---

## ✅ STEP 12: Test from Your Mac (2 minutes)

**Open a NEW terminal on your Mac and test:**

```bash
# Test health endpoint
curl http://35.210.251.24:8000/health

# Expected output:
# {"status":"healthy","model_loaded":true,"tokenizer_loaded":true,"device":"cpu","ready":true}

# Test chat endpoint
curl -X POST http://35.210.251.24:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the top 3 places to visit in Istanbul?",
    "max_tokens": 150,
    "temperature": 0.7
  }'

# You should get a response about Istanbul attractions!
```

---

## 🔥 STEP 13: Configure Firewall (if health check fails)

**If the curl commands time out, you need to open port 8000:**

```bash
# On your Mac:
gcloud compute firewall-rules create allow-llm-api \
  --direction=INGRESS \
  --priority=1000 \
  --network=default \
  --action=ALLOW \
  --rules=tcp:8000 \
  --source-ranges=0.0.0.0/0 \
  --description="Allow LLM API access on port 8000"

# Wait 30 seconds, then test again
curl http://35.210.251.24:8000/health
```

---

## 🎉 STEP 14: Run API Server in Background

Once everything works, let's run it in the background:

```bash
# Stop the current server (press Ctrl+C in SSH terminal)

# Run in background with nohup
nohup python llm_api_server.py > llm_api.log 2>&1 &

# Get the process ID
echo $!

# Check logs
tail -f llm_api.log

# Press Ctrl+C to stop watching logs (server keeps running)
```

**To stop the server later:**

```bash
# Find the process
ps aux | grep llm_api_server

# Kill it (replace PID with actual process ID)
kill PID
```

---

## 📊 SUMMARY - What You'll Have Running

```
┌─────────────────────────────────────────────────┐
│  Google Cloud VM: instance-20251109-085407      │
│  IP: 35.210.251.24                              │
│                                                 │
│  ✅ Docker installed                            │
│  ✅ Python 3.11 + venv                          │
│  ✅ Llama 3.1 8B model (16GB)                   │
│  ✅ FastAPI server running on port 8000         │
│                                                 │
│  Endpoints:                                     │
│  • GET  http://35.210.251.24:8000/health       │
│  • POST http://35.210.251.24:8000/generate     │
│  • POST http://35.210.251.24:8000/chat         │
└─────────────────────────────────────────────────┘
```

---

## 🚀 NEXT STEPS (After API is Running)

1. ✅ Integrate with Render backend
2. ✅ Connect Render to Vercel frontend
3. ✅ Test full end-to-end chat flow
4. ✅ Monitor performance and optimize

---

## 💡 QUICK REFERENCE

**Start API server:**
```bash
cd ~/ai-stanbul
source venv/bin/activate
python llm_api_server.py
```

**Run in background:**
```bash
nohup python llm_api_server.py > llm_api.log 2>&1 &
```

**Check logs:**
```bash
tail -f ~/ai-stanbul/llm_api.log
```

**Stop server:**
```bash
pkill -f llm_api_server
```

**Test health:**
```bash
curl http://35.210.251.24:8000/health
```

---

**Status:** 📝 Ready for execution  
**Next Action:** Run Step 5 (reconnect from Mac)  
**Estimated Time:** ~45 minutes total  

**Created:** November 9, 2025
