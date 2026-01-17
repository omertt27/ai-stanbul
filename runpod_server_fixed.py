#!/usr/bin/env python3
"""
FIXED RunPod Server for Llama 3.1 8B (4-bit)
Fixes the empty text generation issue

Deploy this to your RunPod pod to fix the text generation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn
import os
import time
from typing import Optional

# Configuration
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CACHE_DIR = "/workspace/models"
PORT = 8000

# Global model and tokenizer
model = None
tokenizer = None

app = FastAPI(title="Llama 3.1 8B API (FIXED)", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: Optional[str] = None


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    time_taken: float
    model: str


def load_model():
    """Load the model and tokenizer"""
    global model, tokenizer
    
    print("🦙 Loading Llama 3.1 8B (4-bit)...")
    print("=" * 60)
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        token=os.getenv("HF_TOKEN")
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Tokenizer loaded!")
    
    # Load model
    print("📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        cache_dir=CACHE_DIR,
        token=os.getenv("HF_TOKEN"),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    print("✅ Model loaded!")
    
    # Print memory info
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"💾 GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
    
    print("=" * 60)
    print("🎉 Model ready!")


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "quantization": "4-bit"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    if model is None or tokenizer is None:
        return {"status": "error", "message": "Model not loaded", "model_loaded": False}
    
    health_info = {
        "status": "healthy",
        "model": MODEL_ID,
        "model_loaded": True,
        "tokenizer_loaded": True,
        "quantization": "4-bit",
        "device": str(model.device),
        "gpu_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        health_info["gpu_name"] = torch.cuda.get_device_name(0)
        health_info["memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
        health_info["memory_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
    
    return health_info


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    FIXED: Generate text using Llama 3.1 8B
    Properly extracts generated text from model output
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Use prompt directly without applying template
        # Client is responsible for formatting (prompts.py handles this)
        formatted_prompt = request.prompt
        
        # Tokenize
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # FIXED: Properly decode only the generated tokens
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up the text
        generated_text = generated_text.strip()
        
        tokens_generated = len(generated_tokens)
        time_taken = time.time() - start_time
        
        print(f"✅ Generated {tokens_generated} tokens in {time_taken:.2f}s")
        print(f"   Text length: {len(generated_text)} chars")
        if generated_text:
            print(f"   Preview: {generated_text[:100]}...")
        else:
            print(f"   ⚠️  WARNING: Empty text generated!")
        
        return GenerateResponse(
            text=generated_text,
            tokens_generated=tokens_generated,
            time_taken=time_taken,
            model=MODEL_ID
        )
    
    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🚀 Starting FIXED Llama 3.1 8B Server...")
    print(f"📡 Port: {PORT}")
    print(f"🤖 Model: {MODEL_ID}")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )
