# 🚀 RUNPOD SETUP - ULTRA QUICK GUIDE

## 📋 **3 Simple Steps**

### **STEP 1: Get Hugging Face Token** (2 minutes)
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token" → Name it "runpod" → Copy token
3. Looks like: `hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890`

---

### **STEP 2: Open RunPod & Login** (1 minute)
```bash
# In RunPod terminal
pip install -q huggingface-hub
huggingface-cli login --token hf_YOUR_TOKEN_HERE
```

---

### **STEP 3: Install Everything** (5 minutes)

**Copy-paste this ONE command:**

```bash
apt-get update -qq && \
apt-get install -y -qq python3-pip git curl && \
pip3 install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
pip3 install -q transformers accelerate bitsandbytes scipy fastapi uvicorn pydantic && \
cat > llm_api_server_4bit.py << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import uvicorn

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
    print("🚀 Loading Llama 3.1 8B (4-bit)...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, token=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Loaded! GPU: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/generate")
async def generate(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    system = "You are AI Istanbul, an expert travel assistant for Istanbul, Turkey."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": request.prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=request.max_length, 
                            temperature=request.temperature, do_sample=True, top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id)
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
echo "✅ Setup complete! Starting server..." && \
python3 llm_api_server_4bit.py
```

**That's it! Wait 5 minutes for model download.**

---

## ✅ **Test It** (New Terminal)

```bash
# Local test
curl localhost:8888/health

# Internet test (use YOUR proxy URL)
curl https://4vq1b984pitw8s-8888.proxy.runpod.net/health
```

---

## 🔗 **Connect to Backend**

**Render Dashboard → Environment:**
```
LLM_API_URL=https://4vq1b984pitw8s-8888.proxy.runpod.net
ML_SERVICE_ENABLED=false
```

**Save → Redeploy → DONE!** 🎉

---

## 🐛 **Troubleshooting**

| Problem | Solution |
|---------|----------|
| "401 Unauthorized" | Re-run: `huggingface-cli login --token YOUR_TOKEN` |
| "CUDA out of memory" | Change `max_length: int = 128` in file |
| Model download slow | Normal! 8GB takes 3-10 min |
| Can't paste? | Try: Right-click → Paste or Ctrl+Shift+V |

---

## 📚 **More Help?**

- **Detailed Guide:** `RUNPOD_COPY_PASTE_GUIDE.md`
- **Full Setup:** `RUNPOD_5_MINUTE_SETUP.md`
- **Backend Integration:** `RUNPOD_QUICK_START.md`
