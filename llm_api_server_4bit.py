from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import uvicorn
import os

app = FastAPI(title="AI Istanbul LLM API (Llama 3.1 8B 4-bit)")

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
    if not os.getenv("HF_TOKEN") and not os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
        print("⚠️  WARNING: No Hugging Face token found!")
        print("   Run: huggingface-cli login --token YOUR_TOKEN")
        print("   Get token from: https://huggingface.co/settings/tokens")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model with HF token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=True  # Uses saved HF token from huggingface-cli login
    )
    
    # Load tokenizer with HF token
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
    print("🚀 Starting AI Istanbul LLM API Server")
    print("📍 Model: Llama 3.1 8B (4-bit quantization)")
    print("📍 Port: 8888")
    print("🔐 Using Hugging Face token from cache")
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")
