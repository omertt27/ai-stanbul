#!/bin/bash
# Quick fix for AttributeError: 'dict' object has no attribute 'model_type'
# Run this on your RunPod instance to recreate the server with the fix
# 🔥 NOW WITH PreTrainedTokenizerFast FIX (use_fast=False)

echo "🔧 Creating fixed LLM server with use_fast=False..."

cd /workspace || exit 1

cat > /workspace/llm_server.py << 'ENDOFFILE'
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn

app = FastAPI()

# Model configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_PATH = "/workspace/models/Meta-Llama-3.1-8B-Instruct"

# Load model and tokenizer
print("Loading tokenizer...")
# ✅ Don't pass config to tokenizer - it doesn't need it!
# 🔥 CRITICAL: use_fast=False fixes PreTrainedTokenizerFast error
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    use_fast=False  # 🔥 Fixes 'dict' object has no attribute 'model_type' error
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True
)

print("✅ Model loaded successfully!")

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def root():
    return {"status": "LLM Server is running", "model": MODEL_NAME}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Format the prompt for Llama 3.1 Instruct
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": request.message}
        ]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                top_p=0.9
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        return ChatResponse(response=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
ENDOFFILE

echo "✅ Fixed server created at /workspace/llm_server.py"
echo ""
echo "To start the server, run:"
echo "  python3 /workspace/llm_server.py"
echo ""
echo "Or to run in background:"
echo "  nohup python3 /workspace/llm_server.py > server.log 2>&1 &"
