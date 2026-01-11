"""
RunPod Server for Llama 3.1 8B (4-bit quantized)
This server provides OpenAI-compatible API for the Llama model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn
import os
import time
from typing import Optional, List

# Configuration
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CACHE_DIR = "/workspace/models"
PORT = int(os.getenv("PORT", 19123))

# Global model and tokenizer
model = None
tokenizer = None

app = FastAPI(title="Llama 3.1 8B API", version="1.0.0")

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
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
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
    
    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠️  Warning: HF_TOKEN not set. Using cached model if available.")
    
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
        token=hf_token
    )
    print("✅ Tokenizer loaded!")
    
    # Load model
    print("📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        cache_dir=CACHE_DIR,
        token=hf_token,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    print("✅ Model loaded!")
    
    # Print info
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"💾 GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
    
    print("=" * 60)
    print("🎉 Model ready!")
    print()


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
        "quantization": "4-bit",
        "gpu_available": torch.cuda.is_available()
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    if model is None or tokenizer is None:
        return {"status": "error", "message": "Model not loaded"}
    
    health_info = {
        "status": "healthy",
        "model": MODEL_ID,
        "quantization": "4-bit",
        "device": str(model.device),
        "gpu_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        health_info["gpu_name"] = torch.cuda.get_device_name(0)
        health_info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
        health_info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
    
    return health_info


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text using Llama 3.1 8B
    Compatible with our backend's RunPod client
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Build prompt with system message if provided
        if request.system_prompt:
            full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{request.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{request.prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            full_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{request.prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        tokens_generated = outputs.shape[1] - input_length
        time_taken = time.time() - start_time
        
        return GenerateResponse(
            text=generated_text,
            tokens_generated=tokens_generated,
            time_taken=time_taken,
            model=MODEL_ID
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def openai_completions(request: GenerateRequest):
    """
    OpenAI-compatible completions endpoint
    """
    result = await generate(request)
    
    return {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "text": result.text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,  # Not tracked in this simple implementation
            "completion_tokens": result.tokens_generated,
            "total_tokens": result.tokens_generated
        }
    }


if __name__ == "__main__":
    print(f"🚀 Starting Llama 3.1 8B API server on port {PORT}...")
    print(f"📡 API will be available at: http://0.0.0.0:{PORT}")
    print()
    print("Endpoints:")
    print(f"  - GET  /           - Health check")
    print(f"  - GET  /health     - Detailed health")
    print(f"  - POST /generate   - Generate text (our format)")
    print(f"  - POST /v1/completions - OpenAI-compatible")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)
