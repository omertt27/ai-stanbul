# 🚀 CLEAN VERSION - LLM Server (No Emoji, 5 Parts)

The compact version has encoding issues. Let's create it in 5 clean parts with no special characters:

---

## **STEP 1: Remove old file**

```bash
rm -f llm_api_server_4bit.py
```

---

## **PART 1: Imports and Setup**

```bash
cat > llm_api_server_4bit.py << 'EOF1'
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
EOF1
```

---

## **PART 2: Model Loading**

```bash
cat >> llm_api_server_4bit.py << 'EOF2'

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print("Loading Llama 3.1 8B...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
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
    print(f"Model loaded! GPU: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
EOF2
```

---

## **PART 3: Health Endpoint**

```bash
cat >> llm_api_server_4bit.py << 'EOF3'

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
EOF3
```

---

## **PART 4: Generate Endpoint**

```bash
cat >> llm_api_server_4bit.py << 'EOF4'

@app.post("/generate")
async def generate(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    system = "You are AI Istanbul, an expert travel assistant for Istanbul, Turkey."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": request.prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    return {"response": response}
EOF4
```

---

## **PART 5: Istanbul Query and Main**

```bash
cat >> llm_api_server_4bit.py << 'EOF5'

@app.post("/istanbul-query")
async def istanbul_query(request: GenerateRequest):
    return await generate(request)

if __name__ == "__main__":
    print("AI Istanbul LLM Server Starting...")
    uvicorn.run(app, host="0.0.0.0", port=8888)
EOF5
```

---

## **VERIFY**

```bash
wc -l llm_api_server_4bit.py
```

Expected: ~55 lines

```bash
python3 -m py_compile llm_api_server_4bit.py && echo "OK"
```

Expected: `OK`

---

## **START SERVER**

```bash
python3 llm_api_server_4bit.py
```

Expected:
```
AI Istanbul LLM Server Starting...
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
Loading Llama 3.1 8B...
Model loaded! GPU: 5.3 GB
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8888
```

---

## **TEST**

In a NEW terminal:

```bash
curl http://localhost:8888/health
```

Expected: `{"status":"healthy","model_loaded":true}`

🚀 **No emoji, no encoding issues!**
