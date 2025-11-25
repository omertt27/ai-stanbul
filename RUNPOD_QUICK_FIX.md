# 🚀 RunPod LLM Server - Quick Fix (4 Commands)

## The Error You're Getting
```
AttributeError: 'dict' object has no attribute 'model_type'
```

## ✅ The Fix That Works (Copy & Paste These 4 Commands)

> **Root Cause:** Transformers 4.57.0 has a bug where `use_fast=False` doesn't work properly. Downgrade to 4.44.0!

> **Root Cause:** Transformers 4.57.0 has a bug where `use_fast=False` doesn't work properly. Downgrade to 4.44.0!

### Step 1: Downgrade transformers to 4.44.0
```bash
pip install transformers==4.44.0
```

### Step 2: Remove old server file
```bash
rm -f /workspace/llm_server.py
```

### Step 3: Create new server with fix
```bash
cat > /workspace/llm_server.py << 'EOF_COMPLETE'
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn

app = FastAPI()
MODEL_PATH = "/workspace/models/Meta-Llama-3.1-8B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    use_fast=False  # 🔥 CRITICAL FIX for PreTrainedTokenizerFast error
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True
)
print("✅ Model loaded!")

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def root():
    return {"status": "LLM Server is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": request.message}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=request.max_tokens, temperature=request.temperature, do_sample=True, top_p=0.9)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF_COMPLETE
```

### Step 4: Start server
```bash
python3 /workspace/llm_server.py
```

**Expected output:**
```
Loading tokenizer...
Loading model...
Loading checkpoint shards: 100%|████████████████| 4/4 [01:09<00:00, 17.48s/it]
✅ Model loaded!
INFO:     Started server process [4317]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

✅ **Server is now running!**

---

## 🧪 Alternative: Try Without Downgrading First

If you want to try `use_fast=False` with your current transformers version first (may not work on 4.57.0+):

### Alt Step 1: Remove old server file
```bash
rm -f /workspace/llm_server.py
```

### Alt Step 2: Create server (without downgrading)
```bash
cat > /workspace/llm_server.py << 'EOF_NO_DOWNGRADE'
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn

app = FastAPI()
MODEL_PATH = "/workspace/models/Meta-Llama-3.1-8B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    use_fast=False  # 🔥 CRITICAL FIX for PreTrainedTokenizerFast error
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True
)
print("✅ Model loaded!")

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def root():
    return {"status": "LLM Server is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": request.message}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=request.max_tokens, temperature=request.temperature, do_sample=True, top_p=0.9)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF_NO_DOWNGRADE
```

### Alt Step 3: Start server
```bash
python3 /workspace/llm_server.py
```

**Note:** This likely won't work on transformers 4.57.0+. If it fails, use the main fix (Steps 1-4) above.

---

## 🎯 What You Should See

```
Loading tokenizer...
Loading model...
Loading checkpoint shards: 100%|████████████████| 4/4 [01:09<00:00, 17.48s/it]
✅ Model loaded!
INFO:     Started server process [4317]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 🧪 Test It (From Your Mac)

```bash
# Replace YOUR_RUNPOD_IP with your actual RunPod IP
curl http://YOUR_RUNPOD_IP:8000/health
```

Expected response:
```json
{"status":"healthy"}
```

## What Was Wrong?

**The bug:** Transformers 4.57.0 has a regression where `use_fast=False` doesn't properly bypass `PreTrainedTokenizerFast`, causing the `'dict' object has no attribute 'model_type'` error.

**The solution:** Downgrade to transformers 4.44.0, which correctly handles `use_fast=False` and falls back to the slow tokenizer.

---

**Need more details?** See the complete guide: `RUNPOD_TOKENIZER_FIX.md`
