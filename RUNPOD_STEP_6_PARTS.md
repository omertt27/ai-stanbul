# 🚀 STEP 6: Create LLM Server (In 4 Parts)

The file is too long for one command, so we'll create it in 4 smaller parts:

---

## **PART 1: Create File with Imports and Setup**

```bash
cat > llm_api_server_4bit.py << 'PART1'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import uvicorn
import os

app = FastAPI(title="AI Istanbul LLM API (4-bit)")

model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 256
    temperature: float = 0.7
PART1
```

**✅ Press ENTER after pasting**

---

## **PART 2: Add Model Loading Function**

```bash
cat >> llm_api_server_4bit.py << 'PART2'

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print("🚀 Loading Llama 3.1 8B with 4-bit quantization...")
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
PART2
```

**✅ Press ENTER after pasting**

---

## **PART 3: Add Model Initialization**

```bash
cat >> llm_api_server_4bit.py << 'PART3'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded! GPU: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "gpu_memory_gb": f"{torch.cuda.memory_allocated()/1024**3:.1f}" if torch.cuda.is_available() else "N/A"
    }
PART3
```

**✅ Press ENTER after pasting**

---

## **PART 4A: Add Generate Function (First Half)**

```bash
cat >> llm_api_server_4bit.py << 'PART4A'

@app.post("/generate")
async def generate(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    system_prompt = """You are AI Istanbul, an expert travel assistant for Istanbul, Turkey.
Provide helpful, accurate, and concise information about Istanbul's attractions, restaurants, 
culture, transportation, and hidden gems. Include coordinates when mentioning locations."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.prompt}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
PART4A
```

**✅ Press ENTER after pasting**

---

## **PART 4B: Add Generate Function (Second Half)**

```bash
cat >> llm_api_server_4bit.py << 'PART4B'
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_length,
        temperature=request.temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    
    return {"response": response, "model": "meta-llama/Llama-3.1-8B-Instruct"}
PART4B
```

**✅ Press ENTER after pasting**

---

## **PART 4C: Add Istanbul Query and Main**

```bash
cat >> llm_api_server_4bit.py << 'PART4C'

@app.post("/istanbul-query")
async def istanbul_query(request: GenerateRequest):
    return await generate(request)

if __name__ == "__main__":
    print("🚀 AI Istanbul LLM API Server (Llama 3.1 8B 4-bit)")
    print("📍 Port: 8888")
    print("🔐 Using Hugging Face token from cache")
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")
PART4C
```

**✅ Press ENTER after pasting**

---

## **VERIFY: Check File Was Created**

```bash
wc -l llm_api_server_4bit.py
```

**Expected:** `~95 llm_api_server_4bit.py`

```bash
head -10 llm_api_server_4bit.py
```

**Expected:** Should show the imports

```bash
tail -5 llm_api_server_4bit.py
```

**Expected:** Should show the uvicorn.run line

---

## **START THE SERVER**

```bash
python3 llm_api_server_4bit.py
```

**Now wait 5-10 minutes for model download!**

---

## 🎉 **Expected Output:**

```
🚀 AI Istanbul LLM API Server (Llama 3.1 8B 4-bit)
📍 Port: 8888
🔐 Using Hugging Face token from cache
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
🚀 Loading Llama 3.1 8B with 4-bit quantization...
Downloading shards: 100%|██████████| 8.0G/8.0G
Loading checkpoint shards: 100%|██████████| 4/4
✅ Model loaded! GPU: 5.8 GB
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8888
```

---

## ✅ **Quick Copy-Paste All 3 Parts:**

If you want to paste all at once (might work):

```bash
cat > llm_api_server_4bit.py << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import uvicorn
import os

app = FastAPI(title="AI Istanbul LLM")
model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 256
    temperature: float = 0.7

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print("🚀 Loading Llama 3.1 8B...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True, token=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Loaded! GPU: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None, "model": "meta-llama/Llama-3.1-8B-Instruct"}

@app.post("/generate")
async def generate(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    system = "You are AI Istanbul, an expert travel assistant for Istanbul, Turkey."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": request.prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=request.max_length, temperature=request.temperature, do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    return {"response": response}

@app.post("/istanbul-query")
async def istanbul_query(request: GenerateRequest):
    return await generate(request)

if __name__ == "__main__":
    print("🚀 AI Istanbul LLM (Llama 3.1 8B 4-bit)")
    uvicorn.run(app, host="0.0.0.0", port=8888)
EOF
```

**Then start:**
```bash
python3 llm_api_server_4bit.py
```
